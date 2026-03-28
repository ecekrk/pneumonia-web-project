# 🫁 Pnömoni Tespiti Web Uygulaması (EfficientNet-B3)

Bu proje, göğüs röntgeni (X-ray) görüntülerinden **pnömoni (zatürre) tespiti** yapan, derin öğrenme tabanlı bir web uygulamasıdır. Model olarak **EfficientNet-B3** kullanılmış ve uygulama **Streamlit** ile kullanıcı dostu bir arayüz üzerinden sunulmuştur.

---

## 🚀 Proje Özellikleri

* 🔍 Göğüs röntgeni görüntülerinden pnömoni tespiti
* ⚡ Gerçek zamanlı tahmin ve olasılık çıktısı
* 🎯 Threshold (eşik değeri) ile daha kontrollü karar mekanizması
* 📊 Model performans dashboard’u (Accuracy, Precision, Recall, F1, ROC-AUC vb.)
* 📈 Eğitim süreci grafikleri (Loss ve Accuracy)
* 🌐 Streamlit ile deploy edilebilir web uygulaması

---

## 🧠 Model Bilgileri

* **Model:** EfficientNet-B3
* **Görev:** İkili sınıflandırma (PNEUMONIA / NORMAL)
* **Framework:** PyTorch

### Eğitim Detayları

* Loss Function: CrossEntropyLoss
* Optimizer: Adam

### Kullanılan Performans Metrikleri

* Accuracy
* Precision
* Recall
* F1-Score
* ROC-AUC
* Cohen’s Kappa
* MAE

---

## 📊 Model Performansı (Test Seti)

| Metrik    | Değer  |
| --------- | ------ |
| Accuracy  | 0.8702 |
| Precision | 0.8323 |
| Recall    | 0.9923 |
| F1-Score  | 0.9053 |
| ROC-AUC   | 0.9465 |

📌 Modelin **Recall değerinin yüksek olması**, pnömoni vakalarının kaçırılmaması açısından kritik öneme sahiptir.

---

## 📂 Kullanılan Veri Seti

Bu projede aşağıdaki Kaggle veri seti kullanılmıştır:

👉 [https://www.kaggle.com/code/adinishad/chest-x-ray-images-pneumonia](https://www.kaggle.com/code/adinishad/chest-x-ray-images-pneumonia)

### Sınıflar:

* NORMAL
* PNEUMONIA

---

## 🖥️ Uygulama Yapısı

Proje, çok sayfalı bir Streamlit uygulaması olarak tasarlanmıştır:

* **Proje Özeti:** Projenin amacı ve genel açıklamalar
* **Model Performansı:** Metrikler, ROC Curve, Confusion Matrix ve eğitim grafikleri
* **Canlı Tahmin:** Kullanıcının yüklediği görüntü üzerinden model tahmini

---

## 📁 Proje Yapısı

```bash
PNEUMONIA_WEB_PROJECT/
│
├── streamlit_app/        # Web arayüzü
├── src/                  # Model ve tahmin kodları
├── outputs/              # Grafikler ve metrikler
├── saved_models/         # Eğitilmiş model
├── requirements.txt      # Bağımlılıklar
```

---

## ⚙️ Kurulum ve Çalıştırma

### 1. Ortam oluştur

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
```

### 2. Bağımlılıkları yükle

```bash
pip install -r requirements.txt
```

### 3. Uygulamayı başlat

```bash
streamlit run streamlit_app/app.py
```

---

## 🌐 Canlı Demo

https://pneumonia-web-projectgit-bco4isffyngzwjbxrjm8vk.streamlit.app/

---

## 📌 Notlar

* Model, tıbbi karar destek sistemi olarak tasarlanmıştır ancak **tek başına klinik kullanım için yeterli değildir**
* Amaç: Yapay zekâ destekli sağlık uygulamalarına örnek bir sistem geliştirmektir

---
