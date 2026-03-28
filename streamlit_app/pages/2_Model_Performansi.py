import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "streamlit_app"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from styles import get_custom_css
from utils import load_summary_metrics, get_figure_path, load_and_pad_image

st.set_page_config(page_title="Model Performansı", page_icon="📊", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("📊 Model Performansı")

st.markdown(
    """
    <div class="info-card">
        <h3>Performans Değerlendirmesi</h3>
        <p>
            Bu sayfada modelin test verisi üzerindeki performans sonuçları sunulmaktadır.
            Accuracy, precision, recall, F1-score, MAE, Cohen’s Kappa ve ROC-AUC gibi metrikler
            birlikte değerlendirilerek modelin genel başarımı yorumlanmıştır.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

metrics = load_summary_metrics()

if metrics is None:
    st.error("Özet metrikler bulunamadı. Önce evaluate.py çalıştırılmalı.")
    st.stop()


def metric_box(title, value, description):
    st.markdown(
        f"""
        <div class="metric-card">
            <h4 style="margin-bottom:10px;">{title}</h4>
            <div style="font-size:2rem; font-weight:800; margin-bottom:10px;">
                {value}
            </div>
            <p style="margin:0; opacity:0.92;">
                {description}
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )


col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    metric_box(
        "Accuracy",
        metrics.get("accuracy", "-"),
        "Modelin tüm tahminler içindeki doğruluk oranını gösterir (TP + TN) / (TP + TN + FP + FN)."
    )

with col2:
    metric_box(
        "Precision",
        metrics.get("precision", "-"),
        "Pnömoni olarak tahmin edilenlerin ne kadarının gerçekten pnömoni olduğunu gösterir: TP / (TP + FP)."
    )

with col3:
    metric_box(
        "Recall",
        metrics.get("recall", "-"),
        "Gerçek pnömoni vakalarının ne kadarının doğru yakalandığını gösterir: TP / (TP + FN)."
    )

with col4:
    metric_box(
        "F1-Score",
        metrics.get("f1_score", "-"),
        "Precision ve recall dengesini gösteren harmonik ortalamadır: 2TP / (2TP + FP + FN)."
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4, gap="medium")

with col5:
    metric_box(
        "MAE",
        metrics.get("mae", "-"),
        "Tahmin ile gerçek değer arasındaki ortalama hata miktarını ölçer (yanlış sınıflandırma oranına karşılık gelir)."
    )

with col6:
    metric_box(
        "Cohen's Kappa",
        metrics.get("cohen_kappa", "-"),
        "Modelin rastgele tahmine göre ne kadar daha iyi olduğunu ölçer (1’e yaklaştıkça daha iyi)."
    )

with col7:
    metric_box(
        "ROC-AUC",
        metrics.get("roc_auc", "-"),
        "Modelin farklı eşiklerde sınıfları ayırt etme başarısını gösterir (1’e yakın değerler daha iyidir)."
    )

with col8:
    metric_box(
        "Specificity",
        metrics.get("specificity", "-"),
        "Sağlıklı (NORMAL) vakaların doğru şekilde tanınma oranını gösterir: TN / (TN + FP)."
    )

st.markdown("### Seçilen Threshold")
st.markdown(
    f"""
    <div class="info-card" style="padding:18px 22px;">
        <p style="margin:0;">
            <b>Best Threshold:</b> {metrics.get('best_threshold', '-')}
        </p>
        <p style="margin-top:8px;">
            Threshold, modelin çıktısını sınıflandırma kararına dönüştürmek için kullanılan bir eşik değeridir. Bu eşik değeri, modelin pnömoni olasılığına göre daha dengeli ve kontrollü karar vermesini sağlamak amacıyla seçilmiştir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# TÜM GRAFİKLER İÇİN AYNI BOYUT
GRAPH_SIZE = (1200, 650)

st.markdown("### Confusion Matrix")
cm_path = get_figure_path("efficientnet_b3_confusion_matrix.png")
if cm_path.exists():
    cm_img = load_and_pad_image(cm_path, target_size=GRAPH_SIZE)
    st.image(cm_img, use_container_width=True)
else:
    st.warning("Confusion matrix görseli bulunamadı.")

st.markdown("### Normalized Confusion Matrix")
norm_cm_path = BASE_DIR / "outputs" / "figures" / "efficientnet_b3_confusion_matrix_normalized.png"

if norm_cm_path.exists():
    st.image(str(norm_cm_path), use_container_width=True)
else:
    st.warning("Normalized confusion matrix bulunamadı.")

st.markdown("### ROC Curve")
roc_path = get_figure_path("efficientnet_b3_roc_curve.png")
if roc_path.exists():
    roc_img = load_and_pad_image(roc_path, target_size=GRAPH_SIZE)
    st.image(roc_img, use_container_width=True)
else:
    st.warning("ROC curve görseli bulunamadı.")

st.markdown("### Precision-Recall Curve")
pr_curve_path = BASE_DIR / "outputs" / "figures" / "efficientnet_b3_pr_curve.png"

if pr_curve_path.exists():
    st.image(str(pr_curve_path), use_container_width=True)
else:
    st.warning("Precision-Recall curve bulunamadı.")
st.markdown(
    """
    Precision-Recall eğrisi, özellikle sınıf dengesizliği bulunan veri setlerinde model performansını daha doğru analiz etmek için kullanılır.
    """
)

st.markdown("### Eğitim Grafikleri")

loss_path = get_figure_path("efficientnet_b3_loss_curve.png")
if loss_path.exists():
    st.markdown("#### Training / Validation Loss")
    loss_img = load_and_pad_image(loss_path, target_size=GRAPH_SIZE)
    st.image(loss_img, use_container_width=True)
else:
    st.warning("Loss grafiği bulunamadı.")

acc_path = get_figure_path("efficientnet_b3_accuracy_curve.png")
if acc_path.exists():
    st.markdown("#### Training / Validation Accuracy")
    acc_img = load_and_pad_image(acc_path, target_size=GRAPH_SIZE)
    st.image(acc_img, use_container_width=True)
else:
    st.warning("Accuracy grafiği bulunamadı.")