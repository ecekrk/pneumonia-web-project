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

st.markdown(
    """
    <div class="hero-card">
        <h2>Pnömoni Tespiti için Derin Öğrenme Tabanlı Web Sistemi</h2>
        <p>
            Bu projede akciğer röntgen görüntülerinden pnömoni tespiti yapılmıştır.
            Amaç, derin öğrenme tabanlı bir model ile görüntünün NORMAL veya PNEUMONIA olarak
            sınıflandırılması ve sonucun modern bir web arayüzü üzerinden sunulmasıdır.
        </p>
        <p>
            Uygulama; proje özeti, model performansı ve canlı tahmin olmak üzere üç temel bölümden oluşmaktadır.
            Kullanıcı sol menü üzerinden sayfalar arasında geçiş yaparak hem modelin başarımını inceleyebilir
            hem de yeni bir görüntü yükleyerek anlık tahmin alabilir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <h3>Problem Tanımı</h3>
        <p>
            Pnömoni, özellikle erken tanı gerektiren önemli solunum yolu hastalıklarından biridir.
            Bu çalışmada chest X-ray görüntüleri kullanılarak bir sınıflandırma modeli geliştirilmiş
            ve bu modelin sonuçları web tabanlı bir arayüz üzerinden sunulmuştur.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

col1, col2 = st.columns(2, gap="large")

with col1:
    st.markdown(
        """
        <div class="info-card">
            <h3>Veri Seti</h3>
            <ul>
                <li>Veri seti: Chest X-Ray Images (Pneumonia)</li>
                <li>Sınıflar: NORMAL, PNEUMONIA</li>
                <li>Train: 4434 görüntü</li>
                <li>Validation: 782 görüntü</li>
                <li>Test: 624 görüntü</li>
            </ul>
            <p>
                Veri seti iki sınıflı bir tıbbi görüntü sınıflandırma problemine uygundur ve
                modelin eğitim, doğrulama ve test süreçlerinde kullanılmıştır.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>Kullanılan Model</h3>
            <ul>
                <li>Model: EfficientNet-B3</li>
                <li>Yaklaşım: Transfer Learning</li>
                <li>Girdi boyutu: 224x224</li>
                <li>Loss: Weighted CrossEntropyLoss</li>
                <li>Optimizer: Adam</li>
                <li>Threshold optimizasyonu: Uygulandı</li>
            </ul>
            <p>
                Model, sınıf dengesizliği ve tıbbi hata maliyeti göz önünde bulundurularak
                değerlendirilmiş; özellikle pnömoni vakalarını kaçırmama hedefi ön planda tutulmuştur.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="info-card">
        <h3>Neden EfficientNet-B3 Seçildi?</h3>
        <p>
            EfficientNet-B3, görüntü sınıflandırma problemlerinde güçlü özellik çıkarımı yapabilen,
            transfer learning için uygun bir mimaridir. Sağlık görüntülerinde, basit CNN yapılarına göre
            daha iyi genelleme performansı sağlayabildiği için tercih edilmiştir.
        </p>
        <p>
            Ayrıca modelin karar mekanizmasını daha dengeli hale getirmek için threshold optimizasyonu uygulanmış,
            böylece özellikle yanlış negatiflerin azaltılması hedeflenmiştir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <h3>Sistem Nasıl Çalışır?</h3>
        <p>
            Kullanıcı canlı tahmin sayfasında bir akciğer röntgen görüntüsü yükler.
            Sistem bu görüntüyü ön işleme adımlarından geçirir, EfficientNet-B3 modeli ile analiz eder
            ve NORMAL veya PNEUMONIA tahmini üretir. Sonuç ekranında tahmin sınıfı, güven skoru,
            sınıf olasılıkları ve kullanılan threshold bilgisi gösterilir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)