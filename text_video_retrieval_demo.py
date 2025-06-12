# %%writefile /content/text_video_retrieval_demo.py
import argparse
import os
import json
from pathlib import Path
from ruamel.yaml import YAML

import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np

# Import các thành phần từ codebase ALBEF
from models.model_retrieval import ALBEF
from models.vit import interpolate_pos_embed
from transformers import BertTokenizer # Sử dụng tokenizer chính thức của Hugging Face
import utils # Để sử dụng AttrDict

# Import các thành phần transforms từ dataset/__init__.py
from torchvision import transforms
from dataset.randaugment import RandomAugment # Đảm bảo randaugment.py tồn tại và import được

# Import decord cho việc tải video
try:
    from decord import VideoReader, cpu
except ImportError:
    print("Warning: Decord not found. Please install it (pip install decord) for video loading.")
    VideoReader = None

# --- Hàm tiền xử lý video (lấy từ dataset/__init__.py) ---
def load_and_preprocess_video(video_path, transform, num_frames_per_video, image_res):
    """Tải và lấy mẫu khung hình từ một video, sau đó áp dụng transform."""
    if VideoReader is None:
        print(f"Error: Decord not installed. Cannot load video {video_path}. Returning black frames.")
        return torch.zeros(num_frames_per_video, 3, image_res, image_res)
    
    try:
        vr = VideoReader(video_path, ctx=cpu(0))
        total_frames = len(vr)
        
        if total_frames == 0:
            print(f"Warning: Video {video_path} has 0 frames. Returning black frames.")
            return torch.zeros(num_frames_per_video, 3, image_res, image_res) # Placeholder

        indices = np.linspace(0, total_frames - 1, num_frames_per_video, dtype=int)
        
        frames = []
        for idx in indices:
            frame = Image.fromarray(vr[idx].asnumpy()) 
            frames.append(transform(frame))
        
        return torch.stack(frames) # Shape: (F, C, H, W)
    
    except Exception as e:
        print(f"Error loading video {video_path}: {e}. Returning black frames.")
        return torch.zeros(num_frames_per_video, 3, image_res, image_res)


# --- Hàm chính cho demo ---
def main(args):
    device = torch.device(args.device)

    # 1. Tải cấu hình
    print(f"Loading config from: {args.config}")
    yaml_parser = YAML(typ='safe')
    try:
        with open(args.config, 'r') as f:
            config = yaml_parser.load(f)
    except FileNotFoundError:
        print(f"Error: Config file not found at {args.config}")
        exit(1)
    except Exception as e:
        print(f"Error loading config file: {e}")
        exit(1)

    # Đảm bảo các tham số quan trọng có trong config
    config['num_frames_per_video'] = config.get('num_frames_per_video', 12)
    config['image_res'] = config.get('image_res', 224)

    # 2. Chuẩn bị Transforms cho video
    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))
    video_transform = transforms.Compose([
        transforms.Resize((config['image_res'], config['image_res']), interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
    ])

    # 3. Tải Tokenizer và Model
    print("Loading tokenizer and model...")
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    if args.checkpoint:
        print(f"Loading checkpoint from {args.checkpoint}")
        try:
            checkpoint = torch.load(args.checkpoint, map_location='cpu')
            state_dict = checkpoint['model']

            # Xử lý positional embedding và các key không tương thích
            if 'visual_encoder.pos_embed' in state_dict:
                pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
                state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            if 'visual_encoder_m.pos_embed' in state_dict:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

            # Xử lý các key BERT
            for key in list(state_dict.keys()):
                if 'bert' in key:
                    encoder_key = key.replace('bert.', '')
                    state_dict[encoder_key] = state_dict[key]
                    del state_dict[key]
            
            # Xóa các key không mong muốn hoặc bị thiếu nếu không cần thiết
            # Các key như text_encoder.cls.predictions.* thường là của pretraining heads không dùng trong ALBEF
            # idx_queue có thể bị thiếu nếu checkpoint là từ phiên bản cũ hơn.
            keys_to_remove = [k for k in state_dict.keys() if 'cls.predictions' in k]
            for k in keys_to_remove:
                del state_dict[k]
            if 'idx_queue' in state_dict: # idx_queue là buffer, không phải param
                del state_dict['idx_queue']
            
            # Load state dict, strict=False để bỏ qua các key không khớp hoàn toàn
            msg = model.load_state_dict(state_dict, strict=False)
            print("Checkpoint loaded with message:", msg)

        except Exception as e:
            print(f"Error loading checkpoint {args.checkpoint}: {e}")
            print("Model will start with random weights (or pre-trained HuggingFace weights if applicable).")
    else:
        print("No checkpoint provided. Model will start with random weights (or pre-trained HuggingFace weights if applicable).")


    model.eval() # Chuyển sang chế độ đánh giá
    model = model.to(device)

    print("\n--- Demo Text-Video Retrieval ---")

    # --- Video Candidates ---
    # Ví dụ: lấy 3 video từ MSR-VTT test set
    # Bạn cần đảm bảo các file video này tồn tại ở đường dẫn đã cho
    # Vui lòng thay đổi các đường dẫn này đến các file video thực tế của bạn
    video_root = config['image_root'] # Lấy từ config file
    video_candidates = [
        os.path.join(video_root, 'videoID_1000.mp4'), # Thay bằng ID video thực tế từ test set của bạn
        os.path.join(video_root, 'videoID_1001.mp4'), # Thay bằng ID video thực tế từ test set của bạn
        os.path.join(video_root, 'videoID_1002.mp4'), # Thay bằng ID video thực tế từ test set của bạn
        # Thêm nhiều video hơn nếu muốn
    ]

    # Mã hóa các video ứng cử viên
    print("\nEncoding video candidates...")
    video_embeddings = []
    video_paths = []
    for video_path in video_candidates:
        if not os.path.exists(video_path):
            print(f"Warning: Video file not found: {video_path}. Skipping.")
            continue

        video_tensor = load_and_preprocess_video(
            video_path,
            video_transform,
            config['num_frames_per_video'],
            config['image_res']
        ).unsqueeze(0).to(device) # Thêm batch dimension và chuyển lên device

        with torch.no_grad():
            # Mã hóa video tương tự như trong ALBEF.forward / evaluation
            B_demo, F_demo, C_demo, H_demo, W_demo = video_tensor.shape
            video_flat = video_tensor.view(B_demo * F_demo, C_demo, H_demo, W_demo)
            
            image_embeds_flat = model.visual_encoder(video_flat) 
            image_feat_flat = F.normalize(model.vision_proj(image_embeds_flat[:,0,:]),dim=-1) 
            
            # Average pooling để có biểu diễn video
            video_feat = image_feat_flat.view(B_demo, F_demo, -1).mean(dim=1) # (B, D)
            video_embeddings.append(video_feat)
            video_paths.append(video_path)
    
    if not video_embeddings:
        print("No valid video candidates found. Exiting demo.")
        return

    video_embeddings_tensor = torch.cat(video_embeddings, dim=0) # (num_candidates, embed_dim)
    print(f"Encoded {len(video_embeddings)} video candidates.")

    # --- Text Query ---
    query = "A person playing a musical instrument." # Thay đổi truy vấn của bạn ở đây
    print(f"\nText Query: \"{query}\"")

    # Mã hóa truy vấn văn bản
    text_input = tokenizer(query, padding='longest', max_length=30, return_tensors="pt").to(device)
    with torch.no_grad():
        text_output = model.text_encoder(text_input.input_ids, attention_mask = text_input.attention_mask, mode='text')
        text_embedding = F.normalize(model.text_proj(text_output.last_hidden_state[:,0,:]),dim=-1) # (1, embed_dim)

    # --- Tính toán độ tương đồng và sắp xếp ---
    print("Calculating similarities and ranking videos...")
    similarities = (text_embedding @ video_embeddings_tensor.t()).squeeze(0) # (num_candidates,)

    ranked_indices = torch.argsort(similarities, descending=True)

    print("\n--- Ranked Videos (Highest Similarity First) ---")
    for i, idx in enumerate(ranked_indices):
        sim = similarities[idx].item()
        path = video_paths[idx]
        print(f"Rank {i+1}: Video: {os.path.basename(path)}, Similarity: {sim:.4f}")

    print("\nDemo finished.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='./configs/Retrieval_msrvtt.yaml', help='Path to config file')
    parser.add_argument('--checkpoint', default='/content/ALBEF.pth', help='Path to ALBEF checkpoint. Leave empty to run without checkpoint.')
    parser.add_argument('--text_encoder', default='bert-base-uncased', help='Text encoder name')
    parser.add_argument('--device', default='cuda', help='Device to use (cuda or cpu)')
    args = parser.parse_args()

    main(args)