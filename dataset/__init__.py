# %%writefile /content/ALBEF/dataset/__init__.py
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

# Xóa các import cũ không cần thiết từ caption_dataset nếu không sử dụng cho các tác vụ khác
# Nếu `pretrain_dataset` vẫn được dùng, giữ lại nó hoặc di chuyển định nghĩa của nó vào đây.
# Giả sử bạn vẫn cần `pretrain_dataset` nên giữ import này.
from dataset.caption_dataset import pretrain_dataset
from dataset.nlvr_dataset import nlvr_dataset
from dataset.ve_dataset import ve_dataset
from dataset.vqa_dataset import vqa_dataset
from dataset.grounding_dataset import grounding_dataset

from dataset.randaugment import RandomAugment

# Cần thêm các imports này ở đầu file
import os
import json # Để đọc file annotation JSON cho MSR-VTT
import numpy as np
from torch.utils.data import Dataset
# Import decord
try:
    from decord import VideoReader, cpu
    # Bạn có thể uncomment dòng dưới nếu muốn decord trả về tensor torch trực tiếp
    # decord.bridge.set_bridge("torch")
except ImportError:
    print("Warning: Decord not found. Please install it (pip install decord) for video loading. Placeholder will be used if not installed.")
    VideoReader = None # Placeholder if decord is not available


# --- Đảm bảo các transforms đã được định nghĩa ---
# Giữ nguyên như ALBEF gốc, nhưng lưu ý các transform này áp dụng cho TỪNG khung hình
normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

# image_res được lấy từ config, nhưng ở đây chúng ta dùng 224 hardcode để đảm bảo đúng.
# Trong hàm create_dataset dưới đây, chúng ta sẽ truyền config['image_res'] vào.
# Các transform này sẽ được tạo trong create_dataset để có thể sử dụng config['image_res']
# Đây là cách chính xác hơn:
# pretrain_transform = transforms.Compose([...])
# train_transform = transforms.Compose([...])
# test_transform = transforms.Compose([...])


# --- Định nghĩa các lớp Dataset mới cho Video Retrieval ---

class VideoRetrievalTrainDataset(Dataset):
    def __init__(self, ann_file, transform, video_root, num_frames_per_video):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.video_root = video_root
        self.num_frames_per_video = num_frames_per_video

        # Trích xuất văn bản và video_ids từ annotation
        self.text = [entry['caption'] for entry in self.ann]
        self.video_ids = [entry['video_id'] for entry in self.ann]

    def __len__(self):
        return len(self.ann)

    def _load_video_frames(self, video_path):
        """Tải và lấy mẫu khung hình từ một video."""
        if VideoReader is None:
            print(f"Error: Decord not installed. Cannot load video {video_path}. Returning black frames.")
            return torch.zeros(self.num_frames_per_video, 3, 224, 224) # Hardcoded 224,224 for safety if image_res not available

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)

            if total_frames == 0:
                print(f"Warning: Video {video_path} has 0 frames. Returning black frames.")
                return torch.zeros(self.num_frames_per_video, 3, 224, 224) # Placeholder

            indices = np.linspace(0, total_frames - 1, self.num_frames_per_video, dtype=int)

            frames = []
            for idx in indices:
                frame = Image.fromarray(vr[idx].asnumpy())
                frames.append(self.transform(frame))

            return torch.stack(frames)

        except Exception as e:
            print(f"Error loading video {video_path}: {e}. Returning black frames.")
            return torch.zeros(self.num_frames_per_video, 3, 224, 224)

    def __getitem__(self, index):
        ann_entry = self.ann[index]
        caption = ann_entry['caption']
        video_id = ann_entry['video_id']

        video_path = os.path.join(self.video_root, f"{video_id}.mp4")

        video_tensor = self._load_video_frames(video_path)

        return video_tensor, caption, index


class VideoRetrievalEvalDataset(Dataset):
    def __init__(self, ann_file, transform, video_root, num_frames_per_video):
        self.ann = json.load(open(ann_file, 'r'))
        self.transform = transform
        self.video_root = video_root
        self.num_frames_per_video = num_frames_per_video

        self.text = [ann['caption'] for ann in self.ann]
        self.video_ids = [ann['video_id'] for ann in self.ann]

        self.txt2img = list(range(len(self.text)))
        self.img2txt = [[i] for i in range(len(self.video_ids))]

    def __len__(self):
        return len(self.ann)

    def _load_video_frames(self, video_path):
        if VideoReader is None:
            print(f"Error: Decord not installed. Cannot load video {video_path}. Returning black frames.")
            return torch.zeros(self.num_frames_per_video, 3, 224, 224)

        try:
            vr = VideoReader(video_path, ctx=cpu(0))
            total_frames = len(vr)
            if total_frames == 0:
                print(f"Warning: Video {video_path} has 0 frames. Returning black frames.")
                return torch.zeros(self.num_frames_per_video, 3, 224, 224)
            indices = np.linspace(0, total_frames - 1, self.num_frames_per_video, dtype=int)
            frames = []
            for idx in indices:
                frame = Image.fromarray(vr[idx].asnumpy())
                frames.append(self.transform(frame))
            return torch.stack(frames)
        except Exception as e:
            print(f"Error loading video {video_path}: {e}. Returning black frames.")
            return torch.zeros(self.num_frames_per_video, 3, 224, 224)

    def __getitem__(self, index):
        ann_entry = self.ann[index]
        video_id = ann_entry['video_id']
        video_path = os.path.join(self.video_root, f"{video_id}.mp4")

        video_tensor = self._load_video_frames(video_path)

        return video_tensor, index


# --- Sửa đổi hàm create_dataset để sử dụng các lớp dataset mới ---
def create_dataset(dataset, config):

    normalize = transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711))

    # Transforms được tạo ở đây để có thể sử dụng config['image_res']
    pretrain_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'],scale=(0.2, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0), interpolation=Image.BICUBIC),
            transforms.RandomHorizontalFlip(),
            RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                              'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
        transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
        transforms.ToTensor(),
        normalize,
        ])

    if dataset=='pretrain':
        dataset = pretrain_dataset(config['train_file'], pretrain_transform)
        return dataset

    elif dataset=='re':
        # Sử dụng các lớp dataset video mới
        train_dataset = VideoRetrievalTrainDataset(config['train_file'][0], train_transform, config['image_root'], config['num_frames_per_video'])
        val_dataset = VideoRetrievalEvalDataset(config['val_file'], test_transform, config['image_root'], config['num_frames_per_video'])
        test_dataset = VideoRetrievalEvalDataset(config['test_file'], test_transform, config['image_root'], config['num_frames_per_video'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='vqa':
        train_dataset = vqa_dataset(config['train_file'], train_transform, config['vqa_root'], config['vg_root'], split='train')
        vqa_test_dataset = vqa_dataset(config['test_file'], test_transform, config['vqa_root'], config['vg_root'], split='test', answer_list=config['answer_list'])
        return train_dataset, vqa_test_dataset

    elif dataset=='nlvr':
        train_dataset = nlvr_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = nlvr_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = nlvr_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='ve':
        train_dataset = ve_dataset(config['train_file'], train_transform, config['image_root'])
        val_dataset = ve_dataset(config['val_file'], test_transform, config['image_root'])
        test_dataset = ve_dataset(config['test_file'], test_transform, config['image_root'])
        return train_dataset, val_dataset, test_dataset

    elif dataset=='grounding':
        train_transform = transforms.Compose([
                transforms.Resize((config['image_res'],config['image_res']),interpolation=Image.BICUBIC),
                transforms.RandomHorizontalFlip(),
                RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
                                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
                transforms.ToTensor(),
                normalize,
            ])
        train_dataset = grounding_dataset(config['train_file'], train_transform, config['image_root'], mode='train')
        test_dataset = grounding_dataset(config['test_file'], test_transform, config['image_root'], mode='test')
        return train_dataset, test_dataset

def vqa_collate_fn(batch):
    image_list, question_list, answer_list, weight_list, n = [], [], [], [], []
    for image, question, answer, weights in batch:
        image_list.append(image)
        question_list.append(question)
        weight_list += weights
        answer_list += answer
        n.append(len(answer))
    return torch.stack(image_list,dim=0), question_list, answer_list, torch.Tensor(weight_list), n


def create_sampler(datasets, shuffles, num_tasks, global_rank):
    samplers = []
    for dataset,shuffle in zip(datasets,shuffles):
        sampler = torch.utils.data.DistributedSampler(dataset, num_replicas=num_tasks, rank=global_rank, shuffle=shuffle)
        samplers.append(sampler)
    return samplers


def create_loader(datasets, samplers, batch_size, num_workers, is_trains, collate_fns):
    loaders = []
    for dataset,sampler,bs,n_worker,is_train,collate_fn in zip(datasets,samplers,batch_size,num_workers,is_trains,collate_fns):
        if is_train:
            shuffle = (sampler is None)
            drop_last = True
        else:
            shuffle = False
            drop_last = False
        loader = DataLoader(
            dataset,
            batch_size=bs,
            num_workers=n_worker,
            pin_memory=True,
            sampler=sampler,
            shuffle=shuffle,
            collate_fn=collate_fn,
            drop_last=drop_last,
        )
        loaders.append(loader)
    return loaders