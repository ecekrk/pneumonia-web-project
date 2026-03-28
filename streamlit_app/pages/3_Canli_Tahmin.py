import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import plotly.express as px
import plotly.graph_objects as go

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

PLOTLY_TEMPLATE = "plotly_dark"

def style_plot(fig, height=420):
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(l=30, r=30, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(size=15),
        title=dict(font=dict(size=22)),
        legend=dict(font=dict(size=13))
    )
    return fig

# -------------------------
# SESSION STATE (CANLI VERİ)
# -------------------------
if "live_history" not in st.session_state:
    st.session_state.live_history = []

# -------------------------
# HEADER
# -------------------------
st.markdown(
    """
    <div class="hero-card">
        <h2>Akciğer Röntgen Görüntüsü ile Pnömoni Tahmini</h2>
        <p>
            Bu sayfada yüklenen görüntüler model tarafından analiz edilir ve
            tahminler canlı olarak istatistiksel olarak takip edilir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# UPLOAD UI
# -------------------------
st.markdown(
    """
    <div class="upload-box">
        <div class="upload-title">📤 Görüntü Yükle</div>
        <div class="upload-sub">
            JPG, JPEG veya PNG formatında bir akciğer röntgen görüntüsü yükleyin
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader(
    "",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed"
)

# -------------------------
# TAHMİN
# -------------------------
if uploaded_file is not None:
    result, saved_path = run_prediction_on_uploaded_file(uploaded_file)

    # 🔥 HISTORY'YE EKLE
    st.session_state.live_history.append({
        "pred": result["predicted_class"],
        "confidence": result["confidence_score"],
        "normal_prob": result["probabilities"]["NORMAL"],
        "pneumonia_prob": result["probabilities"]["PNEUMONIA"]
    })

    if len(st.session_state.live_history) > 200:
        st.session_state.live_history.pop(0)

    col1, col2 = st.columns([1, 1], gap="large")

    # -------------------------
    # GÖRÜNTÜ
    # -------------------------
    with col1:
        st.markdown("### Yüklenen Görüntü")
        st.image(saved_path, use_container_width=True)

    # -------------------------
    # SONUÇ
    # -------------------------
    with col2:
        st.markdown("### Tahmin Sonucu")

        if result["predicted_class"] == "PNEUMONIA":
            st.error(f"Pnömoni (%{result['confidence_score']})")
        else:
            st.success(f"Normal (%{result['confidence_score']})")

# -------------------------
# CANLI ANALİTİK PANELİ
# -------------------------
if len(st.session_state.live_history) > 0:

    st.markdown("---")
    st.subheader("📊 Canlı Model Davranışı Analizi")

    df = pd.DataFrame(st.session_state.live_history)

    # -------------------------
    # METRİKLER
    # -------------------------
    col1, col2, col3 = st.columns(3)

    avg_conf = df["confidence"].mean()

    with col1:
        st.metric("Ortalama Confidence", f"%{avg_conf:.2f}")

    # ENTROPY
    def entropy(row):
        probs = np.array([row["normal_prob"], row["pneumonia_prob"]]) / 100
        return -np.sum(probs * np.log(probs + 1e-9))

    df["entropy"] = df.apply(entropy, axis=1)

    with col2:
        st.metric("Ortalama Entropy", round(df["entropy"].mean(), 4))

    # CONFIDENCE MARGIN
    df["margin"] = abs(df["normal_prob"] - df["pneumonia_prob"])

    with col3:
        st.metric("Confidence Margin", f"%{df['margin'].mean():.2f}")

    st.markdown("### Belirsizlik ve Ayrım Gücü")

    col_e1, col_e2 = st.columns(2, gap="large")

    with col_e1:
        fig_entropy = px.line(
            df.reset_index(),
            x=df.reset_index().index,
            y="entropy",
            markers=True,
            title="Entropy Değişimi"
        )
        fig_entropy.update_layout(xaxis_title="Tahmin Sırası", yaxis_title="Entropy")
        style_plot(fig_entropy, height=360)
        st.plotly_chart(fig_entropy, use_container_width=True)

        st.markdown(
            """
            <div class="info-card">
                Entropy, modelin tahminlerindeki belirsizlik seviyesini ölçer; düşük entropy değeri modelin daha net karar verdiğini gösterir.
            </div>
            """,
            unsafe_allow_html=True
        )

    with col_e2:
        fig_margin = px.line(
            df.reset_index(),
            x=df.reset_index().index,
            y="margin",
            markers=True,
            title="Confidence Margin Değişimi"
        )
        fig_margin.update_layout(xaxis_title="Tahmin Sırası", yaxis_title="Margin (%)")
        style_plot(fig_margin, height=360)
        st.plotly_chart(fig_margin, use_container_width=True)

        st.markdown(
            """
            <div class="info-card">
                Confidence margin, iki sınıf arasındaki olasılık farkını gösterir; yüksek değerler modelin daha net ayrım yaptığını ifade eder.
            </div>
            """,
            unsafe_allow_html=True
        )

    # -------------------------
    # CLASS DISTRIBUTION
    # -------------------------
    st.markdown("### Sınıf Dağılımı")
    class_counts = df["pred"].value_counts().reset_index()
    class_counts.columns = ["Sınıf", "Tahmin Sayısı"]

    fig_class = px.bar(
        class_counts,
        x="Sınıf",
        y="Tahmin Sayısı",
        text="Tahmin Sayısı",
        title="Tahmin Edilen Sınıfların Dağılımı"
    )
    fig_class.update_traces(textposition="outside")
    style_plot(fig_class)
    st.plotly_chart(fig_class, use_container_width=True)

    st.markdown(
        """
        <div class="info-card">
            Modelin yaptığı tahminlerin sınıflara göre dağılımını göstererek hangi sınıfa daha fazla eğilim gösterdiğini ortaya koyar.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("### Confidence Dağılımı")
    fig_conf = px.histogram(
        df,
        x="confidence",
        nbins=12,
        title="Confidence Skorlarının Dağılımı",
        labels={"confidence": "Confidence (%)", "count": "Frekans"}
    )
    style_plot(fig_conf)
    st.plotly_chart(fig_conf, use_container_width=True)

    st.markdown(
        """
        <div class="info-card">
            Modelin tahminlerindeki güven skorlarının dağılımını gösterir; yüksek değerler modelin kararlarından daha emin olduğunu ifade eder.
        </div>
        """,
        unsafe_allow_html=True
    )

    # -------------------------
    # DATAFRAME
    # -------------------------
    st.markdown("### Son Tahminler")
    st.dataframe(
        df.tail(10).reset_index(drop=True),
        use_container_width=True,
        height=320
    )

else:
    st.info("Canlı analiz için en az 1 tahmin yapılmalıdır.")