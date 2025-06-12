# %%writefile /content/ALBEF/prepare_msrvtt_data.py
import json
import os
from sklearn.model_selection import train_test_split # Cần cài đặt: pip install scikit-learn
import random

# --- Cấu hình Đường dẫn ---
# Thay đổi đường dẫn này để khớp với vị trí thư mục MSRVTT của bạn
MSRVTT_ROOT = '/content/drive/MyDrive/Intern_FPT/AI/DATASET/MSRVTT/'

FULL_ANNOTATION_FILE = os.path.join(MSRVTT_ROOT, 'annotation', 'MSR_VTT.json')
TRAIN_VIDEO_IDS_FILE = os.path.join(MSRVTT_ROOT, 'high-quality/structured-symlinks', 'test_list_mini.txt')
TEST_VIDEO_IDS_FILE = os.path.join(MSRVTT_ROOT, 'high-quality/structured-symlinks', 'train_list_full_first20.txt')

OUTPUT_ANNOTATION_DIR = os.path.join(MSRVTT_ROOT, 'annotation')
OUTPUT_TRAIN_FILE = os.path.join(OUTPUT_ANNOTATION_DIR, 'msrvtt_train.json')
OUTPUT_VAL_FILE = os.path.join(OUTPUT_ANNOTATION_DIR, 'msrvtt_val.json')
OUTPUT_TEST_FILE = os.path.join(OUTPUT_ANNOTATION_DIR, 'msrvtt_test.json')

VAL_SPLIT_RATIO = 0.1 # Tỷ lệ phần trăm video từ tập huấn luyện để tạo tập validation

# Đặt một seed ngẫu nhiên để đảm bảo các kết quả phân chia là nhất quán
random.seed(42)

# --- Bước 1: Tải tất cả dữ liệu cần thiết ---
print(f"Đang tải file annotation đầy đủ từ: {FULL_ANNOTATION_FILE}")
with open(FULL_ANNOTATION_FILE, 'r') as f:
    full_ann = json.load(f)

print(f"Đang tải ID video huấn luyện từ: {TRAIN_VIDEO_IDS_FILE}")
with open(TRAIN_VIDEO_IDS_FILE, 'r') as f:
    train_video_ids_raw = [line.strip() for line in f]

print(f"Đang tải ID video kiểm tra từ: {TEST_VIDEO_IDS_FILE}")
with open(TEST_VIDEO_IDS_FILE, 'r') as f:
    test_video_ids_raw = [line.strip() for line in f]

# Chuyển đổi sang set để tra cứu nhanh hơn
train_video_ids_set = set(train_video_ids_raw)
test_video_ids_set = set(test_video_ids_raw)

# --- Bước 2: Tạo ánh xạ từ video_id đến các chú thích của nó ---
print("Đang tạo ánh xạ từ ID video đến các chú thích...")
video_annotations_map = {}
# Duyệt qua phần "annotations" của JSON
for ann_entry in full_ann['annotations']:
    video_id = ann_entry['image_id'] # Trong MSR_VTT.json, 'image_id' chính là ID video
    caption = ann_entry['caption']

    # Định dạng lại entry để khớp với cấu trúc mong muốn của dataset
    formatted_entry = {"video_id": video_id, "caption": caption}

    if video_id not in video_annotations_map:
        video_annotations_map[video_id] = []
    video_annotations_map[video_id].append(formatted_entry)

# --- Bước 3: Đổ dữ liệu vào danh sách ban đầu cho huấn luyện và kiểm tra ---
print("Đang đổ dữ liệu vào danh sách huấn luyện và kiểm tra ban đầu...")

# Thu thập tất cả các ID video duy nhất thuộc tập huấn luyện ban đầu và có chú thích
train_videos_for_split = []
for video_id in train_video_ids_set:
    if video_id in video_annotations_map:
        train_videos_for_split.append(video_id)
    # else:
    #     print(f"Cảnh báo: ID video huấn luyện {video_id} không tìm thấy trong MSR_VTT.json annotations.")

# Thu thập tất cả các chú thích cho tập kiểm tra
test_ann = []
for video_id in test_video_ids_set:
    if video_id in video_annotations_map:
        test_ann.extend(video_annotations_map[video_id])
    # else:
    #     print(f"Cảnh báo: ID video kiểm tra {video_id} không tìm thấy trong MSR_VTT.json annotations.")

# Xáo trộn danh sách ID video huấn luyện để đảm bảo việc phân chia train/val ngẫu nhiên và nhất quán
random.shuffle(train_videos_for_split)

# --- Bước 4: Tạo phân chia train/validation từ các video huấn luyện ---
print(f"Đang phân chia các video huấn luyện (tổng {len(train_videos_for_split)}) thành train/val ({VAL_SPLIT_RATIO} cho val)...")
if VAL_SPLIT_RATIO > 0:
    final_train_video_ids, val_video_ids = train_test_split(
        train_videos_for_split, test_size=VAL_SPLIT_RATIO, random_state=42
    )
else:
    final_train_video_ids = train_videos_for_split
    val_video_ids = []

final_train_ann = []
val_ann = []

for video_id in final_train_video_ids:
    final_train_ann.extend(video_annotations_map[video_id])

for video_id in val_video_ids:
    val_ann.extend(video_annotations_map[video_id])

print(f"Số lượng chú thích tập huấn luyện: {len(final_train_ann)}")
print(f"Số lượng chú thích tập validation: {len(val_ann)}")
print(f"Số lượng chú thích tập kiểm tra: {len(test_ann)}")

# --- Bước 5: Lưu các file annotation mới ---
print(f"Đang lưu các file annotation vào: {OUTPUT_ANNOTATION_DIR}")
os.makedirs(OUTPUT_ANNOTATION_DIR, exist_ok=True)

with open(OUTPUT_TRAIN_FILE, 'w') as f:
    json.dump(final_train_ann, f, indent=4)
print(f"Đã lưu annotation huấn luyện vào {OUTPUT_TRAIN_FILE}")

with open(OUTPUT_VAL_FILE, 'w') as f:
    json.dump(val_ann, f, indent=4)
print(f"Đã lưu annotation validation vào {OUTPUT_VAL_FILE}")

with open(OUTPUT_TEST_FILE, 'w') as f:
    json.dump(test_ann, f, indent=4)
print(f"Đã lưu annotation kiểm tra vào {OUTPUT_TEST_FILE}")

print("Hoàn tất quá trình phân chia dữ liệu!")