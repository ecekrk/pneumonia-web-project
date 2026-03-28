import streamlit as st
import pandas as pd
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "streamlit_app"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from styles import get_custom_css
from utils import run_prediction_on_uploaded_file

st.set_page_config(page_title="Canlı Tahmin", page_icon="🧠", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("🧠 Canlı Tahmin")

st.markdown(
    """
    <div class="info-card">
        Akciğer röntgen görüntüsünü yükleyin. Sistem görüntüyü analiz ederek
        NORMAL veya PNEUMONIA tahmini üretecektir.
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "Bir görüntü seçin",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:
    result, saved_path = run_prediction_on_uploaded_file(uploaded_file)

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("### Yüklenen Görüntü")
        st.image(saved_path, use_container_width=True)

    with col2:
        st.markdown("### Tahmin Sonucu")

        if result["predicted_class"] == "PNEUMONIA":
            st.markdown(
                f"""
                <div class="result-warn">
                    <h3>Tahmin: {result["predicted_class"]}</h3>
                    <p>Güven Skoru: %{result["confidence_score"]}</p>
                    <p>Kullanılan Threshold: {result["threshold_used"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="result-good">
                    <h3>Tahmin: {result["predicted_class"]}</h3>
                    <p>Güven Skoru: %{result["confidence_score"]}</p>
                    <p>Kullanılan Threshold: {result["threshold_used"]}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        st.write("Ham argmax çıktısı:", result["raw_argmax_class"])

        prob_df = pd.DataFrame({
            "Sınıf": list(result["probabilities"].keys()),
            "Olasılık (%)": list(result["probabilities"].values())
        })

        st.markdown("### Sınıf Olasılıkları")
        st.bar_chart(prob_df.set_index("Sınıf"))
        st.dataframe(prob_df, use_container_width=True)