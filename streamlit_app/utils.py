import json
import os
import sys
from pathlib import Path
from PIL import Image, ImageStat
import streamlit as st
import numpy as np


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_image  # noqa: E402
from validate_input import validate_xray_image


def load_summary_metrics():
    metrics_path = BASE_DIR / "outputs" / "metrics" / "efficientnet_b3_summary_metrics.json"
    if metrics_path.exists():
        with open(metrics_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def get_figure_path(filename):
    return BASE_DIR / "outputs" / "figures" / filename


def get_metric_path(filename):
    return BASE_DIR / "outputs" / "metrics" / filename

def validate_xray_image(image_path):
    """
    Yüklenen görselin akciğer röntgenine benzer olup olmadığını kaba kurallarla kontrol eder.
    Bu kontrol kesin tıbbi doğrulama yapmaz; yalnızca alakasız görselleri elemek içindir.
    """
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return False, "Yüklenen dosya geçerli bir görüntü olarak açılamadı."

    width, height = img.size

    # 1) Çok küçük görselleri reddet
    if width < 200 or height < 200:
        return False, "Yüklenen görselin çözünürlüğü çok düşük. Lütfen daha uygun bir akciğer röntgen görüntüsü yükleyin."

    # 2) Görseli gri tonlamaya çevir
    gray = img.convert("L")
    gray_np = np.array(gray)

    # 3) Kontrast çok düşükse reddet
    std_dev = float(np.std(gray_np))
    if std_dev < 15:
        return False, "Görsel kontrastı çok düşük görünüyor. Bu görüntü uygun bir akciğer röntgeni olmayabilir."

    # 4) Çok renkli görüntüleri ele
    rgb_np = np.array(img).astype(np.float32)
    channel_diff = np.mean(np.abs(rgb_np[:, :, 0] - rgb_np[:, :, 1])) + \
                   np.mean(np.abs(rgb_np[:, :, 1] - rgb_np[:, :, 2]))

    if channel_diff > 25:
        return False, "Yüklenen görsel akciğer röntgeni formatına uygun görünmüyor. Lütfen göğüs X-ray görüntüsü yükleyin."

    # 5) En-boy oranı çok uçsa reddet
    aspect_ratio = width / height
    if aspect_ratio < 0.5 or aspect_ratio > 1.8:
        return False, "Görsel oranı akciğer röntgeni için alışılmış formatta görünmüyor."

    return True, "Görsel ön kontrolden geçti."

def run_prediction_on_uploaded_file(uploaded_file):
    uploads_dir = BASE_DIR / "streamlit_app" / "temp_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ✅ YENİ BASİT VALIDATION
    is_valid, message = simple_xray_check(str(file_path))

    if not is_valid:
        return {
            "is_valid": False,
            "validation_message": message
        }, str(file_path)

    # prediction
    result = predict_image(str(file_path))
    result["is_valid"] = True
    result["validation_message"] = "Görsel uygun"

    return result, str(file_path)


def show_metric_card(title, value, help_text=""):
    with st.container(border=True):
        st.markdown(
            f"""
            <div style="padding: 4px 4px 2px 4px;">
                <div style="font-size: 1.1rem; font-weight: 700; margin-bottom: 10px;">
                    {title}
                </div>
                <div style="font-size: 2rem; font-weight: 800; margin-bottom: 4px;">
                    {value}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        st.caption(help_text)


def load_and_pad_image(image_path, target_size=(900, 520), background_color=(255, 255, 255)):
    """
    Görseli bozmadan hedef kutuya sığdırır, eksik kalan alanları beyaz boşlukla doldurur.
    Böylece farklı grafikler Streamlit'te aynı boyutta görünür.
    """
    image = Image.open(image_path).convert("RGB")
    image.thumbnail(target_size)

    canvas = Image.new("RGB", target_size, background_color)

    x = (target_size[0] - image.width) // 2
    y = (target_size[1] - image.height) // 2

    canvas.paste(image, (x, y))
    return canvas


def simple_xray_check(image_path):
    try:
        img = Image.open(image_path).convert("RGB")
        img_np = np.array(img)

        # RGB kanalları arasındaki fark
        r, g, b = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
        diff_rg = np.mean(np.abs(r - g))
        diff_rb = np.mean(np.abs(r - b))
        diff_gb = np.mean(np.abs(g - b))

        avg_diff = (diff_rg + diff_rb + diff_gb) / 3

        # 🔥 threshold (çok kritik)
        if avg_diff < 15:
            return True, "X-ray gibi görünüyor"
        else:
            return False, "Bu görüntü X-ray formatında değil"

    except:
        return False, "Görüntü okunamadı"