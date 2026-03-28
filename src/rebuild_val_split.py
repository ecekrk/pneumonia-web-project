import os
import shutil
import random


random.seed(42)

DATA_DIR = "data/chest_xray"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
VAL_DIR = os.path.join(DATA_DIR, "val")

VAL_RATIO = 0.15  # train içinden %15 validation ayır


def get_image_files(folder):
    valid_exts = {".jpg", ".jpeg", ".png"}
    return [
        f for f in os.listdir(folder)
        if os.path.isfile(os.path.join(folder, f)) and os.path.splitext(f)[1].lower() in valid_exts
    ]


def clear_folder(folder_path):
    if os.path.exists(folder_path):
        shutil.rmtree(folder_path)
    os.makedirs(folder_path, exist_ok=True)


def main():
    print("Validation klasörü yeniden oluşturuluyor...")

    clear_folder(VAL_DIR)

    class_names = [d for d in os.listdir(TRAIN_DIR) if os.path.isdir(os.path.join(TRAIN_DIR, d))]

    for class_name in class_names:
        train_class_dir = os.path.join(TRAIN_DIR, class_name)
        val_class_dir = os.path.join(VAL_DIR, class_name)
        os.makedirs(val_class_dir, exist_ok=True)

        image_files = get_image_files(train_class_dir)
        random.shuffle(image_files)

        val_count = int(len(image_files) * VAL_RATIO)
        val_files = image_files[:val_count]

        for file_name in val_files:
            src_path = os.path.join(train_class_dir, file_name)
            dst_path = os.path.join(val_class_dir, file_name)
            shutil.move(src_path, dst_path)

        print(f"{class_name}: {val_count} görüntü validation'a taşındı.")

    print("Yeni validation split hazır.")


if __name__ == "__main__":
    main()