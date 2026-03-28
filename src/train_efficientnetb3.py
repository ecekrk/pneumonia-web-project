import os
import csv
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import tqdm

from data_loader import get_dataloaders
from model_efficientnetb3 import build_efficientnet_b3


DATA_DIR = "data/chest_xray"
BATCH_SIZE = 16
IMG_SIZE = 224
EPOCHS = 10
LR = 1e-4

MODEL_SAVE_PATH = "saved_models/efficientnet_b3_best.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in tqdm(loader, desc="Training"):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def validate(model, loader, criterion):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Validation"):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    epoch_loss = running_loss / len(loader)
    epoch_acc = 100 * correct / total
    return epoch_loss, epoch_acc


def compute_class_weights(train_loader):
    counts = [0, 0]

    for _, labels in train_loader:
        for label in labels:
            counts[label.item()] += 1

    total = sum(counts)
    weights = [total / c for c in counts]

    return torch.tensor(weights, dtype=torch.float32).to(device), counts


def save_history_to_csv(history, save_path):
    with open(save_path, mode="w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["epoch", "train_loss", "train_acc", "val_loss", "val_acc"])

        for i in range(len(history["train_loss"])):
            writer.writerow([
                i + 1,
                history["train_loss"][i],
                history["train_acc"][i],
                history["val_loss"][i],
                history["val_acc"][i]
            ])


def plot_training_curves(history, output_dir):
    epochs = range(1, len(history["train_loss"]) + 1)

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_loss"], label="Train Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EfficientNet-B3 Training and Validation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficientnet_b3_loss_curve.png"))
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(epochs, history["train_acc"], label="Train Accuracy")
    plt.plot(epochs, history["val_acc"], label="Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy (%)")
    plt.title("EfficientNet-B3 Training and Validation Accuracy")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "efficientnet_b3_accuracy_curve.png"))
    plt.close()


def main():
    os.makedirs("saved_models", exist_ok=True)
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)

    train_loader, val_loader, _, class_names = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )

    print("Sınıflar:", class_names)

    class_weights, class_counts = compute_class_weights(train_loader)
    print("Train class counts:", class_counts)
    print("Class weights:", class_weights)

    model = build_efficientnet_b3(num_classes=len(class_names)).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    history = {
        "train_loss": [],
        "train_acc": [],
        "val_loss": [],
        "val_acc": []
    }

    for epoch in range(EPOCHS):
        print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer)
        val_loss, val_acc = validate(model, val_loader, criterion)

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["val_loss"].append(val_loss)
        history["val_acc"].append(val_acc)

        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), MODEL_SAVE_PATH)
            print("En iyi model kaydedildi.")

    save_history_to_csv(history, "outputs/metrics/efficientnet_b3_training_history.csv")
    plot_training_curves(history, "outputs/figures")

    print("\nEğitim tamamlandı.")
    print(f"En iyi validation accuracy: {best_val_acc:.2f}%")
    print("Training history kaydedildi: outputs/metrics/efficientnet_b3_training_history.csv")
    print("Loss grafiği kaydedildi: outputs/figures/efficientnet_b3_loss_curve.png")
    print("Accuracy grafiği kaydedildi: outputs/figures/efficientnet_b3_accuracy_curve.png")


if __name__ == "__main__":
    main()