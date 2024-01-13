import os
import shutil
import random

data_folder = '/data/disk2/vinhnguyen/Dino/data'
train_folder = '/data/disk2/vinhnguyen/Dino/train'
valid_folder = '/data/disk2/vinhnguyen/Dino/valid'

os.makedirs(train_folder, exist_ok=True)
os.makedirs(valid_folder, exist_ok=True)

all_classes = os.listdir(data_folder)

for class_name in all_classes:
    class_path = os.path.join(data_folder, class_name)
    train_class_path = os.path.join(train_folder, class_name)
    valid_class_path = os.path.join(valid_folder, class_name)

    os.makedirs(train_class_path, exist_ok=True)
    os.makedirs(valid_class_path, exist_ok=True)

    all_files = os.listdir(class_path)
    random.shuffle(all_files)

    split_index = int(0.83 * len(all_files))
    train_files = all_files[:split_index]
    valid_files = all_files[split_index:]

    for file_name in train_files:
        src = os.path.join(class_path, file_name)
        dst = os.path.join(train_class_path, file_name)
        shutil.copy(src, dst)

    for file_name in valid_files:
        src = os.path.join(class_path, file_name)
        dst = os.path.join(valid_class_path, file_name)
        shutil.copy(src, dst)
