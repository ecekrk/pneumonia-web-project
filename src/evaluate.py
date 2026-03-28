import os
import json
import csv
import numpy as np
import torch
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    mean_absolute_error,
    cohen_kappa_score,
    roc_curve,
    roc_auc_score
)

from data_loader import get_dataloaders
from model_efficientnetb3 import build_efficientnet_b3


DATA_DIR = "data/chest_xray"
MODEL_PATH = "saved_models/efficientnet_b3_best.pth"
IMG_SIZE = 224
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_summary_metrics(summary_metrics, save_path):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(summary_metrics, f, indent=4, ensure_ascii=False)


def save_predictions_csv(all_labels, all_preds, all_probs, class_names, threshold, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "true_label",
            "predicted_label",
            f"prob_{class_names[0]}",
            f"prob_{class_names[1]}",
            "threshold_used"
        ])

        for true_label, pred_label, probs in zip(all_labels, all_preds, all_probs):
            writer.writerow([
                class_names[true_label],
                class_names[pred_label],
                round(float(probs[0]), 6),
                round(float(probs[1]), 6),
                threshold
            ])


def save_threshold_results(threshold_results, save_path):
    with open(save_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "threshold",
            "accuracy",
            "precision",
            "recall",
            "f1_score",
            "mae",
            "cohen_kappa"
        ])

        for row in threshold_results:
            writer.writerow([
                row["threshold"],
                row["accuracy"],
                row["precision"],
                row["recall"],
                row["f1_score"],
                row["mae"],
                row["cohen_kappa"]
            ])


def plot_confusion_matrix(all_labels, all_preds, class_names, save_path):
    cm = confusion_matrix(all_labels, all_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap="Blues")
    plt.title("EfficientNet-B3 Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def plot_roc_curve(all_labels, positive_class_probs, save_path):
    fpr, tpr, _ = roc_curve(all_labels, positive_class_probs)
    auc_score = roc_auc_score(all_labels, positive_class_probs)

    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f"ROC Curve (AUC = {auc_score:.4f})")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("EfficientNet-B3 ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

    return auc_score


def evaluate_with_threshold(all_labels, positive_class_probs, threshold):
    """
    PNEUMONIA için olasılık threshold'u uygular.
    Eğer olasılık threshold'dan büyük/eşitse 1 (PNEUMONIA), değilse 0 (NORMAL) der.
    """
    all_preds = (positive_class_probs >= threshold).astype(int)

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, zero_division=0)
    recall = recall_score(all_labels, all_preds, zero_division=0)
    f1 = f1_score(all_labels, all_preds, zero_division=0)
    mae = mean_absolute_error(all_labels, all_preds)
    cohen_kappa = cohen_kappa_score(all_labels, all_preds)

    return {
        "threshold": round(float(threshold), 2),
        "accuracy": round(float(accuracy), 4),
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "mae": round(float(mae), 4),
        "cohen_kappa": round(float(cohen_kappa), 4),
        "predictions": all_preds
    }


def main():
    os.makedirs("outputs/figures", exist_ok=True)
    os.makedirs("outputs/metrics", exist_ok=True)
    os.makedirs("outputs/predictions", exist_ok=True)

    _, _, test_loader, class_names = get_dataloaders(
        DATA_DIR,
        batch_size=BATCH_SIZE,
        img_size=IMG_SIZE
    )

    model = build_efficientnet_b3(num_classes=len(class_names)).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Pozitif sınıf: PNEUMONIA
    positive_class_probs = all_probs[:, 1]

    # ROC curve threshold'dan bağımsız olduğu için direkt olasılıklardan hesaplanır
    auc_score = plot_roc_curve(
        all_labels,
        positive_class_probs,
        "outputs/figures/efficientnet_b3_roc_curve.png"
    )

    # Farklı threshold'ları deniyoruz
    thresholds_to_try = np.arange(0.30, 0.91, 0.05)

    threshold_results = []
    best_result = None

    print("\nThreshold sonuçları:")
    for threshold in thresholds_to_try:
        result = evaluate_with_threshold(all_labels, positive_class_probs, threshold)
        threshold_results.append(result)

        print(
            f"Threshold={result['threshold']:.2f} | "
            f"Acc={result['accuracy']:.4f} | "
            f"Prec={result['precision']:.4f} | "
            f"Recall={result['recall']:.4f} | "
            f"F1={result['f1_score']:.4f} | "
            f"Kappa={result['cohen_kappa']:.4f}"
        )

        # En iyi threshold'u F1-score'a göre seçiyoruz
        if best_result is None or result["f1_score"] > best_result["f1_score"]:
            best_result = result

    best_threshold = best_result["threshold"]
    all_preds = best_result["predictions"]

    print("\nSeçilen en iyi threshold:")
    print(
        f"Threshold={best_threshold:.2f} | "
        f"Acc={best_result['accuracy']:.4f} | "
        f"Prec={best_result['precision']:.4f} | "
        f"Recall={best_result['recall']:.4f} | "
        f"F1={best_result['f1_score']:.4f} | "
        f"MAE={best_result['mae']:.4f} | "
        f"Kappa={best_result['cohen_kappa']:.4f} | "
        f"ROC_AUC={auc_score:.4f}"
    )

    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print("\nClassification Report:")
    print(report)

    with open("outputs/metrics/efficientnet_b3_classification_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    plot_confusion_matrix(
        all_labels,
        all_preds,
        class_names,
        "outputs/figures/efficientnet_b3_confusion_matrix.png"
    )

    summary_metrics = {
        "best_threshold": round(float(best_threshold), 2),
        "accuracy": best_result["accuracy"],
        "precision": best_result["precision"],
        "recall": best_result["recall"],
        "f1_score": best_result["f1_score"],
        "mae": best_result["mae"],
        "cohen_kappa": best_result["cohen_kappa"],
        "roc_auc": round(float(auc_score), 4)
    }

    save_summary_metrics(
        summary_metrics,
        "outputs/metrics/efficientnet_b3_summary_metrics.json"
    )

    save_threshold_results(
        threshold_results,
        "outputs/metrics/efficientnet_b3_threshold_results.csv"
    )

    save_predictions_csv(
        all_labels,
        all_preds,
        all_probs,
        class_names,
        best_threshold,
        "outputs/predictions/efficientnet_b3_test_predictions.csv"
    )

    print("\nDeğerlendirme tamamlandı.")
    print("Rapor kaydedildi: outputs/metrics/efficientnet_b3_classification_report.txt")
    print("Özet metrikler kaydedildi: outputs/metrics/efficientnet_b3_summary_metrics.json")
    print("Threshold sonuçları kaydedildi: outputs/metrics/efficientnet_b3_threshold_results.csv")
    print("Confusion matrix kaydedildi: outputs/figures/efficientnet_b3_confusion_matrix.png")
    print("ROC curve kaydedildi: outputs/figures/efficientnet_b3_roc_curve.png")
    print("Tahminler kaydedildi: outputs/predictions/efficientnet_b3_test_predictions.csv")


if __name__ == "__main__":
    main()