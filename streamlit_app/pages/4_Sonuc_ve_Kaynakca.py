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
st.markdown(
    """
    <div class="info-card">
        <h3>Sonuçların Teknik Değerlendirmesi</h3>
        <p>
            Bu çalışmada geliştirilen EfficientNet-B3 tabanlı model,
            göğüs röntgen görüntülerinden pnömoni tespiti görevinde yüksek performans göstermiştir.
            Modelin <b>recall değerinin 0.99 seviyesinde olması</b>, pnömoni vakalarının büyük ölçüde doğru şekilde tespit edildiğini göstermektedir.
        </p>
        <p>
            Tıbbi uygulamalarda yanlış negatif (hasta olduğu halde sağlıklı tahmin edilmesi) durumları kritik olduğundan,
            modelin yüksek recall değerine sahip olması önemli bir avantajdır.
        </p>
        <p>
            Buna karşın precision değerinin daha düşük olması, modelin bazı durumlarda yanlış pozitif tahminler ürettiğini göstermektedir.
            Bu durum, modelin daha hassas (sensitive) çalışacak şekilde optimize edildiğini ve riskli vakaları kaçırmamak adına
            daha temkinli davrandığını göstermektedir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# MODEL DAVRANIŞI
# -------------------------
st.markdown(
    """
    <div class="info-card">
        <h3>Model Davranışı ve Threshold Etkisi</h3>
        <p>
            Model çıktıları üzerinde yapılan analiz sonucunda, sınıflandırma kararının daha dengeli olması için
            <b>0.9 threshold değeri</b> kullanılmıştır. Bu eşik değeri, modelin pnömoni olasılığına daha yüksek
            güven duymadan pozitif karar vermesini sağlar.
        </p>
        <p>
            Threshold kullanımı sayesinde modelin yanlış pozitif ve yanlış negatif dengesi daha kontrollü hale getirilmiş,
            özellikle tıbbi açıdan kritik olan karar süreçlerinde daha güvenilir sonuçlar elde edilmiştir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# GENEL SONUÇ
# -------------------------
st.markdown(
    """
    <div class="info-card">
        <h3>Genel Sonuç</h3>
        <p>
            Bu proje kapsamında, göğüs röntgen görüntülerinden pnömoni tespiti yapabilen,
            kullanıcı dostu bir web tabanlı karar destek sistemi geliştirilmiştir.
        </p>
        <p>
            Geliştirilen sistem, hem teknik açıdan güçlü bir derin öğrenme modeli içermekte
            hem de kullanıcıların modeli etkileşimli olarak deneyimleyebileceği bir arayüz sunmaktadır.
        </p>
        <p>
            Proje; yapay zekâ, tıbbi görüntü işleme ve web teknolojilerinin birleşimini gösteren
            başarılı bir uygulama örneğidir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# SINIRLAR
# -------------------------
st.markdown(
    """
    <div class="info-card">
        <h3>Sınırlamalar</h3>
        <ul>
            <li>Model yalnızca belirli bir veri seti üzerinde eğitilmiştir</li>
            <li>Gerçek klinik kullanım için ek doğrulama gereklidir</li>
            <li>Farklı veri setleri üzerinde genelleme performansı ayrıca test edilmelidir</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

# -------------------------
# KAYNAKÇA
# -------------------------
st.markdown(
    """
    <div class="info-card">
        <h3>Kaynakça</h3>
        <ul>
            <li>
                Chest X-Ray Images (Pneumonia) Dataset - Kaggle  
                <br>
                https://www.kaggle.com/code/adinishad/chest-x-ray-images-pneumonia
            </li>
            <li>
                Tan, M., & Le, Q. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.
            </li>
            <li>
                PyTorch Documentation  
                https://pytorch.org/docs/
            </li>
            <li>
                Streamlit Documentation  
                https://docs.streamlit.io/
            </li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)