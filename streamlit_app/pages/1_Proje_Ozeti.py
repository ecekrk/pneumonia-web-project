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
            Çalışmanın temel amacı, göğüs röntgeni görüntülerini analiz ederek görüntünün
            <b>NORMAL</b> veya <b>PNEUMONIA</b> olarak sınıflandırılmasını sağlayan bir
            derin öğrenme modeli geliştirmek ve bu modeli web tabanlı bir arayüz üzerinden sunmaktır.
        </p>
        <p>
            Uygulama; proje özeti, model performansı, canlı tahmin ve sonuç/kaynakça bölümlerinden oluşmaktadır.
            Böylece kullanıcı hem geliştirilen sistemin teknik ayrıntılarını inceleyebilmekte
            hem de yeni bir görüntü yükleyerek anlık tahmin alabilmektedir.
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
            Pnömoni, akciğer dokusunu etkileyen ve özellikle erken tanının önemli olduğu ciddi bir solunum yolu hastalığıdır.
            Göğüs röntgen görüntülerinin uzmanlar tarafından değerlendirilmesi zaman alabilmekte ve yoğun klinik ortamlarda
            ek karar destek sistemlerine ihtiyaç duyulabilmektedir. Bu projede, yapay zekâ destekli bir yaklaşım kullanılarak
            pnömoni vakalarının görüntü tabanlı olarak tespit edilmesi hedeflenmiştir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <h3>Projenin Amacı ve Önemi</h3>
        <p>
            Projenin amacı, göğüs röntgeni görüntülerinden pnömoni tespiti yapabilen,
            kullanıcı dostu ve açıklayıcı bir web uygulaması geliştirmektir.
            Bu çalışma; sağlık bilişimi, tıbbi görüntü işleme ve derin öğrenme alanlarının kesişiminde yer almakta olup,
            yapay zekâ tabanlı karar destek sistemlerinin sağlık alanındaki potansiyelini göstermektedir.
        </p>
        <p>
            Özellikle yüksek <b>recall</b> değeri sayesinde pnömoni vakalarının kaçırılmaması hedeflenmiş,
            bu nedenle modelin duyarlılığı klinik bakış açısıyla önemli bir avantaj olarak değerlendirilmiştir.
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
                <li><b>Veri seti:</b> Chest X-Ray Images (Pneumonia)</li>
                <li><b>Sınıflar:</b> NORMAL, PNEUMONIA</li>
                <li><b>Train:</b> 4434 görüntü</li>
                <li><b>Validation:</b> 782 görüntü</li>
                <li><b>Test:</b> 624 görüntü</li>
                <li><b>Toplam:</b> 5840 görüntü</li>
            </ul>
            <p>
                Veri seti iki sınıflı bir tıbbi görüntü sınıflandırma problemine uygundur.
                Veri dağılımında pnömoni sınıfı daha fazla olduğu için sınıf dengesizliği dikkate alınmıştır.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

with col2:
    st.markdown(
        """
        <div class="info-card">
            <h3>Veri Ön İşleme ve Eğitim Ayarları</h3>
            <ul>
                <li><b>Girdi boyutu:</b> 224 × 224</li>
                <li><b>Normalize:</b> Uygulandı</li>
                <li><b>Batch size:</b> 16</li>
                <li><b>Epoch:</b> 10</li>
                <li><b>Learning rate:</b> 1e-4</li>
                <li><b>Augmentation:</b> Kullanılmadı. Çünkü veri seti zaten yeterli çeşitliliğe sahip olup modelin genelleme performansı bu haliyle tatmin edici bulunmuştur.</li>
            </ul>
            <p>
                Görüntüler model girişine uygun hale getirmek için yeniden boyutlandırılmış ve normalize edilmiştir.
                Eğitim, doğrulama ve test ayrımı ayrı klasör yapısı üzerinden gerçekleştirilmiştir.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.markdown(
    """
    <div class="info-card">
        <h3>Kullanılan Model ve Seçim Gerekçesi</h3>
        <p>
            Bu projede <b>EfficientNet-B3</b> modeli kullanılmıştır. EfficientNet-B3,
            görüntü sınıflandırma problemlerinde güçlü özellik çıkarımı yapabilen ve transfer learning yaklaşımıyla
            başarılı sonuçlar verebilen bir derin öğrenme mimarisidir.
        </p>
        <p>
            Basit CNN yapılarıyla karşılaştırıldığında daha güçlü bir temsil kapasitesine sahip olması,
            tıbbi görüntülerde daha iyi genelleme potansiyeli sunması ve hazır ön eğitimli ağırlıklarla çalışabilmesi
            nedeniyle tercih edilmiştir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="info-card">
        <h3>Model Mimarisi ve Eğitim Süreci</h3>
        <p>
            EfficientNet-B3 modeli transfer learning yaklaşımıyla kullanılmış, son sınıflandırma katmanı
            ikili sınıflandırma problemine uygun olacak şekilde yeniden düzenlenmiştir.
            Eğitim sürecinde <b>Adam optimizer</b> ve <b>CrossEntropyLoss</b> kullanılmıştır.
            Ayrıca sınıf dengesizliğinin etkisini azaltmak amacıyla ağırlıklı kayıp yaklaşımı dikkate alınmıştır.
        </p>
        <p>
            Eğitim sonrasında model, test seti üzerinde değerlendirilmiş ve daha dengeli tahminler elde etmek için
            karar mekanizmasına <b>threshold optimizasyonu</b> uygulanmıştır.
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
            ve görüntü için NORMAL veya PNEUMONIA tahmini üretir.
            Sonuç ekranında tahmin sınıfı, güven skoru, sınıf olasılıkları ve kullanılan threshold bilgisi gösterilir.
        </p>
    </div>
    """,
    unsafe_allow_html=True
)