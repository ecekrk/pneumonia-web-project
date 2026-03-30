import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "streamlit_app"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from styles import get_custom_css

st.set_page_config(page_title="Proje Özeti", page_icon="📘", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("📘 Proje Özeti")

# ---------------------------------------------------
# HERO
# ---------------------------------------------------
st.markdown(
    """
    <div class="hero-card">
        <h2>Pnömoni Tespiti için Derin Öğrenme Tabanlı Web Sistemi</h2>
        <p>
            Bu projede akciğer röntgen görüntülerinden pnömoni tespiti yapılmıştır.
            Çalışmanın temel amacı, göğüs röntgeni görüntülerini analiz ederek görüntünün
            <b>NORMAL</b> veya <b>PNEUMONIA</b> olarak sınıflandırılmasını sağlayan bir
            derin öğrenme modeli geliştirmek ve bu modeli web tabanlı bir arayüz üzerinden sunmaktır.
        </p>
        <p>
            Uygulama; proje özeti, model performansı, canlı tahmin ve sonuç/kaynakça bölümlerinden oluşmaktadır.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------
# PROBLEM
# ---------------------------------------------------
st.markdown("""
<div class="section-card">

### Problem Tanımı

Pnömoni, akciğer dokusunu etkileyen ve erken tanı gerektiren ciddi bir solunum yolu hastalığıdır.  

Göğüs röntgen görüntülerinin manuel incelenmesi zaman alıcıdır ve özellikle yoğun klinik ortamlarda hata payı artabilmektedir.  

Bu nedenle, hızlı, tutarlı ve otomatik karar verebilen yapay zekâ destekli sistemlere ihtiyaç duyulmaktadır.

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# AMAÇ
# ---------------------------------------------------
st.markdown("""
<div class="section-card">

### Projenin Amacı ve Önemi

Bu çalışmanın amacı, göğüs röntgen görüntülerinden pnömoni tespiti yapabilen  
hızlı ve güvenilir bir yapay zekâ modeli geliştirmektir.

Özellikle yüksek **recall (duyarlılık)** değeri sayesinde pnömoni vakalarının kaçırılmaması hedeflenmiştir.  

Bu durum klinik açıdan kritik bir avantaj sağlayarak erken teşhis sürecine katkı sunmaktadır.

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# DATASET
# ---------------------------------------------------
col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown("""
    <div class="section-card">

    ### Veri Seti

    - **Veri seti:** Chest X-Ray Images (Pneumonia) – Kaggle  
    - **Sınıflar:** NORMAL, PNEUMONIA  
    - **Toplam:** 5840 görüntü  
    - **Train:** 4434 (%75.9)  
    - **Validation:** 782 (%13.4)  
    - **Test:** 624 (%10.7)  

    Veri seti başlangıçta train/validation/test olarak ayrılmıştır.  
    Ancak validation seti çok küçük olduğu için sınıf dağılımı korunarak yeniden dengelenmiştir.

    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="section-card">

    ### Veri Ön İşleme

    - **Resize:** 224 × 224  
    - **Color:** RGB  
    - **Normalization:** [-1, 1]  
    - **Augmentation:** Kullanılmadı  

    Augmentation kullanılmamıştır çünkü veri seti yeterli çeşitliliğe sahiptir ve  
    model bu veri ile yüksek performans göstermiştir.

    </div>
    """, unsafe_allow_html=True)

st.code(
"""
transform = transforms.Compose([

    transforms.Resize((224, 224)),  # tüm görüntüleri modele uygun boyuta getir

    transforms.ToTensor(),  # görüntüyü PyTorch tensor formatına çevir

    transforms.Normalize(
        [0.5, 0.5, 0.5],    # ortalama
        [0.5, 0.5, 0.5]     # standart sapma
    )  # piksel değerlerini normalize ederek eğitimi stabilize et

])
""",
language="python"
)
# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

# ---------------------------------------------------
# MODEL
# ---------------------------------------------------

st.markdown("""
<div class="section-card">

### Model Seçimi

Bu çalışmada **EfficientNet-B3** modeli tercih edilmiştir.  
EfficientNet mimarisi, modelin derinliğini, genişliğini ve giriş çözünürlüğünü
dengeli bir şekilde artıran **compound scaling** yaklaşımına dayanır.

**Tercih edilme nedenleri:**

- Yüksek doğruluk performansı  
- Transfer learning ile güçlü feature extraction  
- Daha az parametre ile yüksek verimlilik  
- Tıbbi görüntülerde ince detayları yakalayabilme  

Pnömoni gibi ince doku farklılıklarının olduğu problemler için
güçlü feature extraction kritik olduğundan bu model seçilmiştir.

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# MODEL MİMARİSİ
# ---------------------------------------------------

st.markdown("""
<div class="section-card">

### Model Mimarisi

EfficientNet-B3, MBConv blokları ile çok ölçekli feature extraction yapar.  
Son katman, iki sınıflı sınıflandırma için yeniden düzenlenmiştir.

- **Backbone:** EfficientNet-B3  
- **Transfer Learning:** ImageNet  
- **Output:** 2 sınıf (NORMAL / PNEUMONIA)  
- **Activation:** Softmax  

Model önce düşük seviyeli (kenar/doku), ardından yüksek seviyeli
(anatomik yapı) özellikleri öğrenir.

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# HYPERPARAMS
# ---------------------------------------------------

st.markdown("""
<div class="section-card">

### Hiperparametreler

- **Batch Size:** 16 → GPU belleği ve stabil eğitim dengesi  
- **Epoch:** 10 → overfitting riskini azaltmak için sınırlı tutuldu  
- **Learning Rate:** 1e-4 → küçük adımlarla stabil öğrenme  
- **Optimizer:** Adam → adaptif öğrenme oranı ile hızlı yakınsama  
- **Loss Function:** CrossEntropyLoss  

Learning rate düşük seçilerek modelin ani dalgalanmalar yerine daha kontrollü öğrenmesi sağlanmıştır.  
Adam optimizer ise her parametre için ayrı öğrenme oranı ayarlayarak eğitim sürecini hızlandırır.

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# TRAINING
# ---------------------------------------------------

st.markdown("""
<div class="section-card">

### Eğitim Süreci

Model, eğitim veri seti üzerinde batch'ler halinde eğitilmiş ve her epoch sonunda
validation seti ile performansı ölçülmüştür.  
En iyi model validation accuracy değerine göre seçilmiştir.

</div>
""", unsafe_allow_html=True)

st.code(
    """
for epoch in range(EPOCHS):

    model.train()  # eğitim modu

    for images, labels in train_loader:

        outputs = model(images)              # forward pass
        loss = criterion(outputs, labels)    # loss hesaplama

        optimizer.zero_grad()  # gradient temizleme
        loss.backward()        # backpropagation
        optimizer.step()       # ağırlık güncelleme

    model.eval()  # validation modu
""",
    language="python"
)

# ---------------------------------------------------
# THRESHOLD
# ---------------------------------------------------
st.markdown("""
<div class="section-card">

### Karar Mekanizması (Threshold)

Model çıktısı olasılık değerleri üretir.  
Nihai karar, belirlenen threshold değerine göre verilir.

Bu eşik değeri, pnömoni vakalarının kaçırılmaması (yüksek recall) için optimize edilmiştir.

</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# SYSTEM
# ---------------------------------------------------
st.markdown("""
<div class="section-card">

### Sistem Nasıl Çalışır?

Kullanıcı görüntüyü yükler → model görüntüyü işler →  
olasılık değerlerini hesaplar → nihai sınıf tahmini oluşturur → sonuç kullanıcıya sunulur.

Sistem, gerçek zamanlı olarak çalışarak hızlı ve anlaşılır bir karar destek mekanizması sağlar.

</div>
""", unsafe_allow_html=True)