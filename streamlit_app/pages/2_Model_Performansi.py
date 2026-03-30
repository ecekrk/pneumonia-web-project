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

# ---------------------------------------------------
# SAYFA GİRİŞİ
# ---------------------------------------------------
st.markdown("""
<div class="section-card">

### Model Performans Değerlendirmesi

Bu bölümde modelin test veri seti üzerindeki performansı çok yönlü olarak analiz edilmiştir.

Model performansı yalnızca accuracy ile değil; **precision, recall, F1-score, ROC-AUC, specificity, MAE ve Cohen’s Kappa**
gibi metriklerle birlikte değerlendirilmiştir.

Özellikle medikal uygulamalarda kritik olan **recall (duyarlılık)** metriği ön planda tutulmuş,
pnömoni vakalarının kaçırılmaması hedeflenmiştir.

Bu sayede modelin yalnızca genel doğruluğu değil, aynı zamanda **klinik açıdan güvenilirliği**
de analiz edilmiştir.

</div>
""", unsafe_allow_html=True)

metrics = load_summary_metrics()

if metrics is None:
    st.error("Özet metrikler bulunamadı. Önce evaluate.py çalıştırılmalıdır.")
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


# ---------------------------------------------------
# METRİK KARTLARI
# ---------------------------------------------------
col1, col2, col3, col4 = st.columns(4, gap="medium")

with col1:
    metric_box(
        "Accuracy",
        metrics.get("accuracy", "-"),
        "Modelin tüm tahminler içindeki genel doğruluk oranını gösterir: (TP + TN) / (TP + TN + FP + FN)."
    )

with col2:
    metric_box(
        "Precision",
        metrics.get("precision", "-"),
        "Pnömoni olarak tahmin edilen örneklerin ne kadarının gerçekten pnömoni olduğunu gösterir: TP / (TP + FP)."
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
        "Precision ve recall dengesini gösteren harmonik ortalamadır; dengesiz veri setlerinde önemli bir metriktir."
    )

st.markdown("<div style='height: 10px;'></div>", unsafe_allow_html=True)

col5, col6, col7, col8 = st.columns(4, gap="medium")

with col5:
    metric_box(
        "MAE",
        metrics.get("mae", "-"),
        "Tahmin ile gerçek değer arasındaki ortalama hata miktarını gösterir; düşük değerler daha iyi performansa işaret eder."
    )

with col6:
    metric_box(
        "Cohen's Kappa",
        metrics.get("cohen_kappa", "-"),
        "Modelin rastgele tahmine göre ne kadar daha iyi olduğunu ölçer; 1’e yaklaştıkça daha güçlü uyum gösterir."
    )

with col7:
    metric_box(
        "ROC-AUC",
        metrics.get("roc_auc", "-"),
        "Modelin farklı eşik değerlerinde sınıfları ayırt etme başarısını gösterir; 1’e yakın değerler daha iyidir."
    )

with col8:
    metric_box(
        "Specificity",
        metrics.get("specificity", "-"),
        "Sağlıklı (NORMAL) vakaların doğru şekilde tanınma oranını gösterir: TN / (TN + FP)."
    )

# ---------------------------------------------------
# THRESHOLD
# ---------------------------------------------------
st.markdown("""
<div class="section-card">

### Karar Mekanizması (Threshold)

Model çıktısı doğrudan sınıf etiketi değil, her sınıf için olasılık değerleri üretmektedir.  
Nihai sınıflandırma kararı, yalnızca en yüksek olasılığa göre değil, belirlenen **threshold** değeri dikkate alınarak verilmiştir.

Threshold, modelin çıktısını sınıflandırma kararına dönüştürmek için kullanılan bir eşik değeridir.  
Bu eşik değeri, modelin pnömoni olasılığına göre daha dengeli ve kontrollü karar vermesini sağlamak amacıyla seçilmiştir.

Özellikle bu projede threshold optimizasyonu, pnömoni vakalarının kaçırılmaması ve yanlış negatiflerin azaltılması amacıyla uygulanmıştır.

**Seçilen en iyi threshold:** {best_threshold}

</div>
""".replace("{best_threshold}", str(metrics.get("best_threshold", "-"))), unsafe_allow_html=True)

# ---------------------------------------------------
# GRAFİK BOYUTU
# ---------------------------------------------------
GRAPH_SIZE = (1200, 650)

# ---------------------------------------------------
# CONFUSION MATRIX
# ---------------------------------------------------
st.markdown("### Confusion Matrix")
cm_path = get_figure_path("efficientnet_b3_confusion_matrix.png")
if cm_path.exists():
    cm_img = load_and_pad_image(cm_path, target_size=GRAPH_SIZE)
    st.image(cm_img, use_container_width=True)
else:
    st.warning("Confusion matrix görseli bulunamadı.")

st.markdown("""
Bu grafik, modelin hangi sınıfları doğru tahmin ettiğini ve hangi hata türlerini yaptığını göstermektedir.  
Özellikle yanlış negatif ve yanlış pozitif değerleri, modelin klinik güvenilirliğini değerlendirmek açısından önemlidir.
""")

# ---------------------------------------------------
# NORMALIZED CONFUSION MATRIX
# ---------------------------------------------------
st.markdown("### Normalized Confusion Matrix")
norm_cm_path = BASE_DIR / "outputs" / "figures" / "efficientnet_b3_confusion_matrix_normalized.png"

if norm_cm_path.exists():
    norm_cm_img = load_and_pad_image(norm_cm_path, target_size=GRAPH_SIZE)
    st.image(norm_cm_img, use_container_width=True)
else:
    st.warning("Normalized confusion matrix bulunamadı.")

st.markdown("""
Normalize edilmiş confusion matrix, her sınıf için doğru ve yanlış tahmin oranlarını daha net gösterir.  
Bu grafik özellikle sınıflar arası başarıyı yüzdesel olarak yorumlamak için kullanışlıdır.
""")

# ---------------------------------------------------
# ROC CURVE
# ---------------------------------------------------
st.markdown("### ROC Curve")
roc_path = get_figure_path("efficientnet_b3_roc_curve.png")
if roc_path.exists():
    roc_img = load_and_pad_image(roc_path, target_size=GRAPH_SIZE)
    st.image(roc_img, use_container_width=True)
else:
    st.warning("ROC curve görseli bulunamadı.")

st.markdown("""
ROC eğrisi, modelin farklı eşik değerlerinde doğru pozitif oranı ile yanlış pozitif oranı arasındaki ilişkiyi gösterir.  
Eğrinin sol üst köşeye yaklaşması ve AUC değerinin yüksek olması, modelin ayırt etme gücünün yüksek olduğunu gösterir.
""")

# ---------------------------------------------------
# PR CURVE
# ---------------------------------------------------
st.markdown("### Precision-Recall Curve")
pr_curve_path = BASE_DIR / "outputs" / "figures" / "efficientnet_b3_pr_curve.png"

if pr_curve_path.exists():
    pr_img = load_and_pad_image(pr_curve_path, target_size=GRAPH_SIZE)
    st.image(pr_img, use_container_width=True)
else:
    st.warning("Precision-Recall curve bulunamadı.")

st.markdown("""
Precision-Recall eğrisi, özellikle sınıf dengesizliği bulunan veri setlerinde model performansını daha doğru analiz etmek için kullanılır.  
Bu grafik, modelin pozitif sınıf üzerindeki duyarlılık ve doğruluk dengesini göstermektedir.
""")

# ---------------------------------------------------
# EĞİTİM GRAFİKLERİ
# ---------------------------------------------------
st.markdown("### Eğitim Grafikleri")

loss_path = get_figure_path("efficientnet_b3_loss_curve.png")
if loss_path.exists():
    st.markdown("#### Training / Validation Loss")
    loss_img = load_and_pad_image(loss_path, target_size=GRAPH_SIZE)
    st.image(loss_img, use_container_width=True)
else:
    st.warning("Loss grafiği bulunamadı.")

st.markdown("""
Loss eğrileri, modelin eğitim sürecinde hatasının nasıl değiştiğini gösterir.  
Train ve validation loss değerlerinin birlikte değerlendirilmesi, öğrenmenin stabil olup olmadığını ve overfitting riskini anlamak için önemlidir.
""")

acc_path = get_figure_path("efficientnet_b3_accuracy_curve.png")
if acc_path.exists():
    st.markdown("#### Training / Validation Accuracy")
    acc_img = load_and_pad_image(acc_path, target_size=GRAPH_SIZE)
    st.image(acc_img, use_container_width=True)
else:
    st.warning("Accuracy grafiği bulunamadı.")

st.markdown("""
Accuracy eğrileri, modelin epoch’lar boyunca doğruluk seviyesindeki değişimi göstermektedir.  
Train ve validation accuracy değerlerinin birbirine yakın seyretmesi, modelin genelleme performansının güçlü olduğuna işaret eder.
""")

# ---------------------------------------------------
# GENEL YORUM
# ---------------------------------------------------
st.markdown("""
<div class="section-card">

### Sonuçların Yorumlanması

Elde edilen performans değerleri, modelin pnömoni tespiti görevinde başarılı olduğunu göstermektedir.  
Özellikle yüksek **recall** değeri, pnömoni vakalarının büyük bölümünün doğru şekilde tespit edildiğini göstermektedir.

Bu durum medikal açıdan kritik öneme sahiptir; çünkü yanlış negatiflerin azaltılması, riskli hastaların gözden kaçmaması açısından önemlidir.  

Bununla birlikte precision, specificity ve confusion matrix birlikte incelendiğinde modelin bazı durumlarda yanlış pozitif üretebildiği anlaşılmaktadır.  
Bu nedenle model, duyarlılığı yüksek tutacak şekilde optimize edilmiş bir karar destek sistemi olarak değerlendirilmektedir.

</div>
""", unsafe_allow_html=True)