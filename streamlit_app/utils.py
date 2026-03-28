import json
import os
import sys
from pathlib import Path
from PIL import Image
import streamlit as st


BASE_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = BASE_DIR / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.append(str(SRC_DIR))

from predict import predict_image  # noqa: E402


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


def run_prediction_on_uploaded_file(uploaded_file):
    uploads_dir = BASE_DIR / "streamlit_app" / "temp_uploads"
    uploads_dir.mkdir(parents=True, exist_ok=True)

    file_path = uploads_dir / uploaded_file.name

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    result = predict_image(str(file_path))
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