import streamlit as st
from pathlib import Path
import sys

BASE_DIR = Path(__file__).resolve().parents[2]
APP_DIR = BASE_DIR / "streamlit_app"

if str(APP_DIR) not in sys.path:
    sys.path.append(str(APP_DIR))

from styles import get_custom_css

st.set_page_config(page_title="Sonuç ve Kaynakça", page_icon="📚", layout="wide")
st.markdown(get_custom_css(), unsafe_allow_html=True)

st.title("📚 Sonuç ve Kaynakça")

# -------------------------
# SONUÇLARIN YORUMLANMASI
# -------------------------
st.markdown("""
<div class="section-card">

### Sonuçların Teknik Değerlendirmesi

Bu çalışmada geliştirilen **EfficientNet-B3 tabanlı derin öğrenme modeli**,  
göğüs röntgen görüntülerinden pnömoni tespiti görevinde yüksek performans göstermiştir.

Modelin özellikle **yüksek recall (~0.99)** değerine sahip olması, pnömoni vakalarının büyük ölçüde doğru şekilde yakalandığını göstermektedir.  
Bu durum, tıbbi uygulamalarda kritik olan **yanlış negatif (False Negative)** hataların minimize edildiğini ifade eder.

Buna karşılık precision değerinin görece daha düşük olması, modelin bazı durumlarda **yanlış pozitif (False Positive)** tahminler ürettiğini göstermektedir.  
Bu durum, modelin daha hassas (sensitive) çalışacak şekilde optimize edildiğini ve riskli vakaları kaçırmamak adına bilinçli olarak tercih edilmiştir.

</div>
""", unsafe_allow_html=True)

# -------------------------
# THRESHOLD
# -------------------------
st.markdown("""
<div class="section-card">

### Model Davranışı ve Threshold Etkisi

Model çıktıları olasılık değerleri şeklinde üretildiğinden, nihai sınıflandırma kararı için bir **eşik değeri (threshold)** kullanılmaktadır.

Bu çalışmada **0.90 threshold değeri** tercih edilmiştir.  
Bu değer, modelin pnömoni sınıfı için daha yüksek güvene ulaşmadan pozitif karar vermemesini sağlar.

Bu yaklaşım sayesinde:
- Yanlış negatif oranı minimize edilir
- Klinik açıdan riskli durumların kaçırılması engellenir
- Model daha kontrollü ve güvenilir kararlar üretir

</div>
""", unsafe_allow_html=True)

# -------------------------
# GENEL SONUÇ
# -------------------------
st.markdown("""
<div class="section-card">

### Genel Değerlendirme

Bu proje kapsamında, göğüs röntgen görüntülerinden pnömoni tespiti yapabilen  
**yapay zekâ destekli web tabanlı bir karar destek sistemi** geliştirilmiştir.

Geliştirilen sistem:
- Güçlü bir derin öğrenme modeli (EfficientNet-B3)
- Kullanıcı dostu arayüz (Streamlit)
- Canlı analiz ve görselleştirme bileşenleri

içermektedir.

Bu yönüyle proje, **yapay zekâ, tıbbi görüntü işleme ve web teknolojilerinin entegre kullanımına** başarılı bir örnek oluşturmaktadır.

</div>
""", unsafe_allow_html=True)

# -------------------------
# SINIRLAR
# -------------------------
st.markdown("""
<div class="section-card">

### Sınırlamalar

- Model yalnızca belirli bir veri seti üzerinde eğitilmiştir  
- Gerçek klinik ortamda kullanılmadan önce ek doğrulama gereklidir  
- Farklı hastane verileri ile genelleme performansı test edilmelidir  
- Veri setindeki sınıf dağılımı model davranışını etkileyebilir  

</div>
""", unsafe_allow_html=True)

# -------------------------
# GELECEK ÇALIŞMALAR
# -------------------------
st.markdown("""
<div class="section-card">

### Gelecek Çalışmalar

- Daha büyük ve çeşitli veri setleri ile modelin yeniden eğitilmesi  
- Çok sınıflı hastalık tespiti (COVID-19, akciğer nodülleri vb.)  
- Gerçek zamanlı klinik entegrasyon  
- Model açıklanabilirliği (Explainable AI) tekniklerinin eklenmesi  

</div>
""", unsafe_allow_html=True)

# -------------------------
# KAYNAKÇA
# -------------------------
st.markdown("""
<div class="section-card">

### Kaynakça

- Chest X-Ray Images (Pneumonia) Dataset. Kaggle.  
  https://www.kaggle.com/code/adinishad/chest-x-ray-images-pneumonia  

- Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.  

- PyTorch Documentation.  
  https://pytorch.org/docs/  

- Streamlit Documentation.  
  https://docs.streamlit.io/  

</div>
""", unsafe_allow_html=True)