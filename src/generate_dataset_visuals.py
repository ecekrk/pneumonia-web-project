import os
import random
from collections import Counter

import matplotlib.pyplot as plt
from PIL import Image


DATA_DIR = "data/chest_xray"
OUTPUT_DIR = "outputs/figures"
random.seed(42)


def ensure_output_dir():
    os.makedirs(OUTPUT_DIR, exist_ok=True)


def count_images(split):
    split_path = os.path.join(DATA_DIR, split)
    class_counts = {}

    for class_name in os.listdir(split_path):
        class_dir = os.path.join(split_path, class_name)
        if os.path.isdir(class_dir):
            image_count = len([
                f for f in os.listdir(class_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png"))
            ])
            class_counts[class_name] = image_count

    return class_counts


def plot_class_distribution():
    splits = ["train", "val", "test"]

    for split in splits:
        class_counts = count_images(split)

        classes = list(class_counts.keys())
        counts = list(class_counts.values())

        plt.figure(figsize=(6, 4))
        plt.bar(classes, counts)
        plt.title(f"{split.upper()} Class Distribution")
        plt.xlabel("Class")
        plt.ylabel("Image Count")
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_DIR, f"{split}_class_distribution.png"))
        plt.close()


def get_random_images(class_dir, n=3):
    image_files = [
        f for f in os.listdir(class_dir)
        if f.lower().endswith((".jpg", ".jpeg", ".png"))
    ]
    return random.sample(image_files, min(n, len(image_files)))


def plot_sample_images(split="train", images_per_class=3):
    split_path = os.path.join(DATA_DIR, split)
    class_names = [
        d for d in os.listdir(split_path)
        if os.path.isdir(os.path.join(split_path, d))
    ]

    rows = len(class_names)
    cols = images_per_class

    plt.figure(figsize=(4 * cols, 4 * rows))

    plot_index = 1
    for class_name in class_names:
        class_dir = os.path.join(split_path, class_name)
        sample_images = get_random_images(class_dir, images_per_class)

        for image_name in sample_images:
            image_path = os.path.join(class_dir, image_name)
            image = Image.open(image_path).convert("RGB")

            plt.subplot(rows, cols, plot_index)
            plt.imshow(image)
            plt.title(class_name)
            plt.axis("off")
            plot_index += 1

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, f"{split}_sample_images.png"))
    plt.close()


def main():
    ensure_output_dir()
    plot_class_distribution()
    plot_sample_images(split="train", images_per_class=3)

    print("Grafikler oluşturuldu:")
    print("- outputs/figures/train_class_distribution.png")
    print("- outputs/figures/val_class_distribution.png")
    print("- outputs/figures/test_class_distribution.png")
    print("- outputs/figures/train_sample_images.png")


if __name__ == "__main__":
    main()