import streamlit as st
import pandas as pd
from pathlib import Path
import sys
import plotly.express as px

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "streamlit_app"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from styles import get_custom_css, get_upload_css
from utils import run_prediction_on_uploaded_file

st.set_page_config(page_title="Canlı Tahmin", page_icon="🧠", layout="wide")

st.markdown(get_custom_css(), unsafe_allow_html=True)
st.markdown(get_upload_css(), unsafe_allow_html=True)

st.title("🧠 Canlı Tahmin")

# -------------------------
# SESSION
# -------------------------
if "live_history" not in st.session_state:
    st.session_state.live_history = []

# -------------------------
# HEADER
# -------------------------
st.markdown("""
<div class="hero-card">
    <h2>Akciğer Röntgen Görüntüsü ile Pnömoni Tahmini</h2>
    <p>
        Bu sayfada yüklenen görüntüler derin öğrenme modeli tarafından analiz edilir.
        Yapılan tahminler yalnızca tekil olarak değil, aynı zamanda canlı istatistiksel analizlerle birlikte değerlendirilir.
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="section-card">

### Canlı Tahmin Sayfası Nasıl Çalışır?

Kullanıcı bir göğüs röntgen görüntüsü yüklediğinde sistem görüntüyü ön işleme adımlarından geçirir,
eğitilmiş EfficientNet-B3 modeli ile analiz eder ve görüntü için **NORMAL** veya **PNEUMONIA** tahmini üretir.

Üretilen tahminler yalnızca sonuç olarak gösterilmez; aynı zamanda modelin confidence ve sınıf dağılımı gibi
davranışları da canlı olarak izlenebilir.

</div>
""", unsafe_allow_html=True)

# -------------------------
# UPLOAD
# -------------------------
st.markdown("""
<div class="upload-box">
    <div class="upload-title">📤 Görüntü Yükle</div>
    <div class="upload-sub">
        JPG / JPEG / PNG formatında röntgen görüntüsü yükleyin
    </div>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

# -------------------------
# PREDICTION
# -------------------------
if uploaded_file is not None:

    result, saved_path = run_prediction_on_uploaded_file(uploaded_file)

    if not result.get("is_valid", False):
        st.image(saved_path, use_container_width=True)
        st.error("Yüklenen görsel bir akciğer röntgeni (X-ray) değildir.")
        st.info("Lütfen uygun bir X-ray görüntüsü yükleyin.")
        st.stop()


    st.session_state.live_history.append({
        "pred": result["predicted_class"],
        "confidence": result["confidence_score"],
        "normal_prob": result["probabilities"]["NORMAL"],
        "pneumonia_prob": result["probabilities"]["PNEUMONIA"]
    })

    if len(st.session_state.live_history) > 200:
        st.session_state.live_history.pop(0)

    col1, col2 = st.columns(2)

    # IMAGE
    with col1:
        st.image(saved_path, use_container_width=True)

    # RESULT BOX
    with col2:
        st.markdown("""
        <div class="section-card">
        <h3>Tahmin Sonucu</h3>
        """, unsafe_allow_html=True)

        if result["predicted_class"] == "PNEUMONIA":
            st.error(f"Pnömoni (%{result['confidence_score']})")
        else:
            st.success(f"Normal (%{result['confidence_score']})")

        st.progress(result["confidence_score"] / 100)

        st.markdown("""
        <p>
        Model güven skoru, tahminin ne kadar güçlü olduğunu gösterir.
        Yüksek değerler modelin kararından daha emin olduğunu ifade eder.
        </p>
        </div>
        """, unsafe_allow_html=True)

# -------------------------
# ANALYTICS
# -------------------------
if len(st.session_state.live_history) > 0:

    st.markdown("---")

    df = pd.DataFrame(st.session_state.live_history)

    # PSEUDO ACCURACY
    pseudo_acc = df["confidence"].mean()

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Pseudo Accuracy", f"%{pseudo_acc:.2f}")

    with col2:
        st.metric("Max Confidence", f"%{df['confidence'].max():.2f}")

    with col3:
        st.metric("Min Confidence", f"%{df['confidence'].min():.2f}")

    st.markdown("""
    <div class="section-card">
    Pseudo accuracy, modelin ortalama güven skoruna göre hesaplanan yaklaşık performans göstergesidir.
    Gerçek etiket bulunmayan durumlarda model davranışını anlamak için kullanılır.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # CLASS DISTRIBUTION
    with col1:
        class_counts = df["pred"].value_counts().reset_index()
        class_counts.columns = ["Sınıf", "Sayı"]

        fig_class = px.bar(class_counts, x="Sınıf", y="Sayı", text="Sayı", title="Sınıf Dağılımı")
        st.plotly_chart(fig_class, use_container_width=True)

        st.markdown("""
        <div class="section-card">
        Bu grafik modelin hangi sınıfa daha fazla tahmin yaptığını gösterir.
        Dengesiz dağılım model biasına işaret edebilir.
        </div>
        """, unsafe_allow_html=True)

    # CONFIDENCE HIST
    with col2:
        fig_conf = px.histogram(df, x="confidence", nbins=10, title="Confidence Dağılımı")
        st.plotly_chart(fig_conf, use_container_width=True)

        st.markdown("""
        <div class="section-card">
        Bu grafik modelin tahminlerindeki güven seviyesinin dağılımını gösterir.
        Yüksek değerler modelin daha emin olduğunu ifade eder.
        </div>
        """, unsafe_allow_html=True)

    # TABLE
    st.markdown("""
    <div class="section-card">
    <h3>Son Tahminler</h3>
    Modelin yaptığı son tahminler aşağıda listelenmiştir.
    </div>
    """, unsafe_allow_html=True)

    st.dataframe(df.tail(10), use_container_width=True)

else:
    st.info("Analiz için görüntü yükleyin.")