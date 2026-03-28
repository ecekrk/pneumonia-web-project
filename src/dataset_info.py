import os
from collections import Counter


DATA_DIR = "data/chest_xray"


def count_images_in_folder(folder_path):
    exts = {".jpg", ".jpeg", ".png"}
    count = 0
    for root, _, files in os.walk(folder_path):
        for file in files:
            if os.path.splitext(file)[1].lower() in exts:
                count += 1
    return count


def count_class_images(split_path):
    class_counts = {}
    for class_name in os.listdir(split_path):
        class_folder = os.path.join(split_path, class_name)
        if os.path.isdir(class_folder):
            class_counts[class_name] = count_images_in_folder(class_folder)
    return class_counts


def main():
    for split in ["train", "val", "test"]:
        split_path = os.path.join(DATA_DIR, split)
        class_counts = count_class_images(split_path)
        total = sum(class_counts.values())

        print(f"\n--- {split.upper()} ---")
        for class_name, count in class_counts.items():
            print(f"{class_name}: {count}")
        print(f"Toplam: {total}")


if __name__ == "__main__":
    main()