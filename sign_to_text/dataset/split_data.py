#sign_to_text/dataset/split_data.py
import os
import random
import shutil

def split_data(base_dir, split_ratio=0.2):
    train_dir = os.path.join(base_dir, 'train')
    val_dir = os.path.join(base_dir, 'val')
    os.makedirs(val_dir, exist_ok=True)

    classes = sorted([d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))])
    print(f"Found classes: {classes}")

    for cls in classes:
        cls_train_path = os.path.join(train_dir, cls)
        cls_val_path = os.path.join(val_dir, cls)
        os.makedirs(cls_val_path, exist_ok=True)

        # Lấy tất cả các thư mục con trong label (nếu có)
        subdirs = [d for d in os.listdir(cls_train_path) if os.path.isdir(os.path.join(cls_train_path, d))]
        if not subdirs:
            # Nếu không có thư mục con thì tìm file trực tiếp trong cls_train_path
            files = [os.path.join(cls_train_path, f) for f in os.listdir(cls_train_path)
                     if f.lower().endswith(('.png','.jpg','.jpeg','.mp4'))]
            if len(files) == 0:
                print(f"Class '{cls}' has 0 images/videos")
                continue
            n_val = int(len(files)*split_ratio)
            val_files = random.sample(files, n_val)
            for f in val_files:
                shutil.move(f, cls_val_path)
            print(f"Class '{cls}' has {len(files)} files")
            print(f"Moved {len(val_files)} files from '{cls_train_path}' to '{cls_val_path}'")
        else:
            # Nếu có thư mục con, xử lý từng thư mục con
            for sub in subdirs:
                sub_train_path = os.path.join(cls_train_path, sub)
                sub_val_path = os.path.join(cls_val_path, sub)
                os.makedirs(sub_val_path, exist_ok=True)
                files = [os.path.join(sub_train_path, f) for f in os.listdir(sub_train_path)
                         if f.lower().endswith(('.png','.jpg','.jpeg','.mp4'))]
                if len(files) == 0:
                    print(f"Subfolder '{sub}' in class '{cls}' has 0 images/videos")
                    continue
                n_val = int(len(files)*split_ratio)
                val_files = random.sample(files, n_val)
                for f in val_files:
                    shutil.move(f, sub_val_path)
                print(f"Subfolder '{sub}' in class '{cls}' has {len(files)} files")
                print(f"Moved {len(val_files)} files from '{sub_train_path}' to '{sub_val_path}'")


if __name__ == '__main__':
    base_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'sign_to_text', 'dataset')
    split_data(base_dir)
