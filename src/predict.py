import os
import json
import torch
from PIL import Image
from torchvision import transforms

from model_efficientnetb3 import build_efficientnet_b3


# =========================================================
# AYARLAR
# =========================================================
MODEL_PATH = "saved_models/efficientnet_b3_best.pth"
METRICS_PATH = "outputs/metrics/efficientnet_b3_summary_metrics.json"
CLASS_NAMES = ["NORMAL", "PNEUMONIA"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# =========================================================
# TRANSFORM
# =========================================================
# Eğitim sırasında 224x224 boyut ve 3 kanallı normalize kullandığımız için
# tahmin aşamasında da aynı ön işleme uygulanmalıdır.
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.5, 0.5, 0.5],
        [0.5, 0.5, 0.5]
    )
])


# =========================================================
# THRESHOLD YÜKLEME
# =========================================================
def load_best_threshold():
    """
    Evaluate aşamasında seçilen en iyi threshold'u JSON dosyasından alır.
    Eğer dosya yoksa güvenli varsayılan olarak 0.50 kullanır.
    """
    if os.path.exists(METRICS_PATH):
        with open(METRICS_PATH, "r", encoding="utf-8") as f:
            metrics = json.load(f)
            return float(metrics.get("best_threshold", 0.50))
    return 0.50


# =========================================================
# MODEL YÜKLEME
# =========================================================
def load_model():
    """
    Eğitilmiş EfficientNet-B3 modelini yükler.
    """
    model = build_efficientnet_b3(num_classes=2).to(device)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    return model


# =========================================================
# TEK GÖRÜNTÜ TAHMİNİ
# =========================================================
def predict_image(image_path):
    """
    Tek bir görüntü için tahmin üretir.

    Dönüş:
    {
        "predicted_class": "...",
        "confidence_score": ...,
        "probabilities": {
            "NORMAL": ...,
            "PNEUMONIA": ...
        },
        "pneumonia_probability": ...,
        "threshold_used": ...,
        "raw_argmax_class": ...
    }
    """
    model = load_model()
    threshold = load_best_threshold()

    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(image_tensor)
        probs = torch.softmax(outputs, dim=1)[0]

    normal_prob = float(probs[0].item())
    pneumonia_prob = float(probs[1].item())

    # Ham argmax sonucu
    raw_argmax_index = torch.argmax(probs).item()
    raw_argmax_class = CLASS_NAMES[raw_argmax_index]

    # Nihai karar threshold'a göre verilir.
    # Çünkü evaluate.py aşamasında gördük ki model argmax ile fazla PNEUMONIA diyebiliyor.
    if pneumonia_prob >= threshold:
        predicted_class = "PNEUMONIA"
        confidence_score = pneumonia_prob * 100
    else:
        predicted_class = "NORMAL"
        confidence_score = normal_prob * 100

    result = {
        "predicted_class": predicted_class,
        "confidence_score": round(confidence_score, 2),
        "probabilities": {
            "NORMAL": round(normal_prob * 100, 2),
            "PNEUMONIA": round(pneumonia_prob * 100, 2)
        },
        "pneumonia_probability": round(pneumonia_prob * 100, 2),
        "threshold_used": threshold,
        "raw_argmax_class": raw_argmax_class
    }

    return result


# =========================================================
# TEST AMAÇLI ÇALIŞTIRMA
# =========================================================
if __name__ == "__main__":
    sample_path = "sample_image.jpeg"

    if os.path.exists(sample_path):
        prediction = predict_image(sample_path)
        print("Tahmin sonucu:")
        print(prediction)
    else:
        print("Test için sample_image.jpeg bulunamadı.")