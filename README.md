# 🧠 Brain Tumor Classification MRI

Bu proje, MRI görüntülerinden beyin tümörü sınıflandırması yapan bir Streamlit web uygulamasıdır. 5 farklı makine öğrenmesi modeli kullanarak tümör türlerini tahmin eder.

## 🎯 Özellikler

- **5 Farklı Model**: CNN, VGG16, MobileNet, SVM, Random Forest
- **4 Tümör Sınıfı**: Glioma, Meningioma, No Tumor, Pituitary
- **Web Arayüzü**: Kullanıcı dostu Streamlit arayüzü
- **Gerçek Zamanlı Tahmin**: Görüntü yükleyip anında sonuç alma

## 📊 Model Performansları

| Model | Accuracy | Size | Type |
|-------|----------|------|------|
| CNN | ~70.65% | 124 KB | Deep Learning |
| VGG16 | ~72.21% | 132 KB | Transfer Learning |
| MobileNet | ~76.88% | 163 KB | Mobile Optimized |
| SVM | ~94.60% | 532 KB | Classical ML |
| Random Forest | ~85.54% | 4.7 KB | Ensemble |

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.8+
- pip

### Kurulum

1. **Projeyi klonlayın:**
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-classification-mri.git
cd brain-tumor-classification-mri
```

2. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **Uygulamayı çalıştırın:**
```bash
streamlit run app.py
```

4. **Tarayıcınızda açın:**
```
http://localhost:8501
```

## 📁 Proje Yapısı

```
beyin_tumoru_app/
├── 📁 models/                    # Eğitilmiş modeller
│   ├── cnn_model.h5             # CNN modeli
│   ├── vgg16_model.h5           # VGG16 modeli  
│   ├── mobilenet_model.h5       # MobileNet modeli
│   ├── svm_model.pkl            # SVM modeli
│   └── rf_model.pkl             # Random Forest modeli
├── 📁 utils/                     # Yardımcı fonksiyonlar
│   ├── __pycache__/             # Python cache
│   ├── preprocessing.py         # Görüntü işleme
│   └── inference_*.py           # Model test kodları
├── 📁 test_images/              # Test görüntüleri
├── 📄 app.py                    # Ana uygulama
├── 📄 requirements.txt          # Gerekli paketler
├── 📄 devcontainer.json         # VS Code container
└── 📄 README.md                 # Proje dokümantasyonu
```

## 🔧 Kullanım

1. **Görüntü Yükleme**: Sol taraftan MRI görüntüsünü yükleyin (JPG, JPEG, PNG)
2. **Model Seçimi**: Dropdown'dan istediğiniz modeli seçin
3. **Tahmin**: Sonuç otomatik olarak görüntülenecek

### Desteklenen Formatlar
- ✅ JPG, JPEG, PNG
- ✅ RGB ve Grayscale görüntüler
- ✅ Herhangi bir boyut (otomatik yeniden boyutlandırılır)

## 🧪 Model Detayları

### CNN (Convolutional Neural Network)
- **Input**: RGB, 224x224
- **Architecture**: Custom CNN
- **Best For**: Genel amaçlı sınıflandırma

### VGG16 (Transfer Learning)
- **Input**: RGB, 224x224  
- **Pre-trained**: ImageNet
- **Best For**: Yüksek doğruluk

### MobileNet (Mobile Optimized)
- **Input**: RGB, 224x224
- **Features**: Lightweight, fast
- **Best For**: Mobil uygulamalar

### SVM (Support Vector Machine)
- **Input**: Grayscale, 224x224 (50,176 features)
- **Kernel**: RBF
- **Best For**: Klasik ML yaklaşımı

### Random Forest
- **Input**: MobileNet features
- **Features**: Deep learning + ensemble
- **Best For**: Güvenilir tahminler

## 📸 Test Görüntüleri

`test_images/` klasöründe örnek MRI görüntüleri bulunmaktadır. Bu görüntüleri kullanarak modelleri test edebilirsiniz.

## 🛠️ Geliştirme

### Yeni Model Ekleme

1. Modelinizi `models/` klasörüne kaydedin
2. `MODEL_PATHS` dictionary'sine ekleyin
3. Gerekirse `preprocessing.py`'de özel işleme ekleyin

### Docker ile Çalıştırma

VS Code Dev Container desteği mevcuttur:
```bash
# .devcontainer/devcontainer.json dosyası kullanılır
# VS Code'da "Reopen in Container" seçeneğini kullanın
```

## 📈 Performans İpuçları

- **En Hızlı**: MobileNet
- **En Doğru**: VGG16
- **En Küçük**: Random Forest
- **En Dengeli**: CNN

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans# 🧠 Brain Tumor Classification MRI

Bu proje, MRI görüntülerinden beyin tümörü sınıflandırması yapan bir Streamlit web uygulamasıdır. 5 farklı makine öğrenmesi modeli kullanarak tümör türlerini tahmin eder.

## 🎯 Özellikler

- **5 Farklı Model**: CNN, VGG16, MobileNet, SVM, Random Forest
- **4 Tümör Sınıfı**: Glioma, Meningioma, No Tumor, Pituitary
- **Web Arayüzü**: Kullanıcı dostu Streamlit arayüzü
- **Gerçek Zamanlı Tahmin**: Görüntü yükleyip anında sonuç alma

## 📊 Model Performansları

| Model | Accuracy | Size | Type |
|-------|----------|------|------|
| CNN | ~95% | 124 MB | Deep Learning |
| VGG16 | ~96% | 132 MB | Transfer Learning |
| MobileNet | ~94% | 163 MB | Mobile Optimized |
| SVM | ~88% | 532 MB | Classical ML |
| Random Forest | ~85% | 4.7 MB | Ensemble |

## 🚀 Hızlı Başlangıç

### Gereksinimler
- Python 3.8+
- pip

### Kurulum

1. **Projeyi klonlayın:**
```bash
git clone https://github.com/YOUR_USERNAME/brain-tumor-classification-mri.git
cd brain-tumor-classification-mri
```

2. **Gerekli paketleri yükleyin:**
```bash
pip install -r requirements.txt
```

3. **⚠️ ÖNEMLİ: Model dosyalarını indirin:**

Model dosyaları boyut sınırları nedeniyle repository'de bulunmaz. İndirmek için:

```bash
# Otomatik model indirme (Önerilen)
python download_models.py
```

**Manuel indirme:**
- Eğer otomatik indirme çalışmazsa, aşağıdaki Google Drive linklerinden modelleri indirip `models/` klasörüne koyun:
  - [cnn_model.h5](https://drive.google.com/uc?id=1fSVLr3PjR7YsUpBKgU4MvqxG9YzvuN6k) (124 MB)
  - [vgg16_model.h5](https://drive.google.com/uc?id=18aoVfmxPaLGV902UulBhQKNrqHEt1TVt) (132 MB)
  - [mobilenet_model.h5](https://drive.google.com/uc?id=1CDPS2wtC8BTFeCWQQn5htIi9lms4ROk9) (163 MB)
  - [svm_model.pkl](https://drive.google.com/uc?id=1PUysW4CWS69HnAOTZTkENOhdi56Bm7dt) (532 MB)
  - [rf_model.pkl](https://drive.google.com/uc?id=1sJq3lUssRlZrxUK6fFkj9U4WhY9z1BRp) (4.7 MB)

4. **Model indirme doğrulaması:**
```bash
# models/ klasöründe aşağıdaki dosyalar olmalı:
ls models/
# Çıktı: cnn_model.h5  mobilenet_model.h5  rf_model.pkl  svm_model.pkl  vgg16_model.h5
```

5. **Uygulamayı çalıştırın:**
```bash
streamlit run app.py
```

6. **Tarayıcınızda açın:**
```
http://localhost:8501
```

## 📁 Proje Yapısı

```
beyin_tumoru_app/
├── 📁 models/                    # Eğitilmiş modeller (indirildikten sonra)
│   ├── cnn_model.h5             # CNN modeli (124 MB)
│   ├── vgg16_model.h5           # VGG16 modeli (132 MB)
│   ├── mobilenet_model.h5       # MobileNet modeli (163 MB)
│   ├── svm_model.pkl            # SVM modeli (532 MB)
│   └── rf_model.pkl             # Random Forest modeli (4.7 MB)
├── 📁 utils/                     # Yardımcı fonksiyonlar
│   ├── __pycache__/             # Python cache
│   ├── preprocessing.py         # Görüntü işleme
│   └── inference_*.py           # Model test kodları
├── 📁 test_images/              # Test görüntüleri
├── 📄 app.py                    # Ana uygulama
├── 📄 download_models.py        # Model indirme scripti ⭐
├── 📄 requirements.txt          # Gerekli paketler
├── 📄 devcontainer.json         # VS Code container
└── 📄 README.md                 # Proje dokümantasyonu
```

## 🔧 Kullanım

### İlk Çalıştırma Kontrolleri

Uygulamayı çalıştırmadan önce:

```bash
# 1. Modellerin indirildiğini kontrol edin
python -c "import os; print('✓ Modeller hazır!' if len(os.listdir('models/')) == 5 else '✗ Modeller eksik, download_models.py çalıştırın')"

# 2. Gerekli paketleri kontrol edin  
python -c "import streamlit, tensorflow, sklearn, cv2; print('✓ Paketler hazır!')"
```

### Uygulama Kullanımı

1. **Görüntü Yükleme**: Sol taraftan MRI görüntüsünü yükleyin (JPG, JPEG, PNG)
2. **Model Seçimi**: Dropdown'dan istediğiniz modeli seçin
3. **Tahmin**: Sonuç otomatik olarak görüntülenecek

### Desteklenen Formatlar
- ✅ JPG, JPEG, PNG
- ✅ RGB ve Grayscale görüntüler
- ✅ Herhangi bir boyut (otomatik yeniden boyutlandırılır)

## 🧪 Model Detayları

### CNN (Convolutional Neural Network)
- **Input**: RGB, 224x224
- **Architecture**: Custom CNN
- **Best For**: Genel amaçlı sınıflandırma

### VGG16 (Transfer Learning)
- **Input**: RGB, 224x224  
- **Pre-trained**: ImageNet
- **Best For**: Yüksek doğruluk

### MobileNet (Mobile Optimized)
- **Input**: RGB, 224x224
- **Features**: Lightweight, fast
- **Best For**: Mobil uygulamalar

### SVM (Support Vector Machine)
- **Input**: Grayscale, 224x224 (50,176 features)
- **Kernel**: RBF
- **Best For**: Klasik ML yaklaşımı

### Random Forest
- **Input**: MobileNet features
- **Features**: Deep learning + ensemble
- **Best For**: Güvenilir tahminler

## 📸 Test Görüntüleri

`test_images/` klasöründe örnek MRI görüntüleri bulunmaktadır. Bu görüntüleri kullanarak modelleri test edebilirsiniz.

## 🛠️ Geliştirme

### Yeni Model Ekleme

1. Modelinizi `models/` klasörüne kaydedin
2. `MODEL_PATHS` dictionary'sine ekleyin
3. Gerekirse `preprocessing.py`'de özel işleme ekleyin

### Docker ile Çalıştırma

VS Code Dev Container desteği mevcuttur:
```bash
# .devcontainer/devcontainer.json dosyası kullanılır
# VS Code'da "Reopen in Container" seçeneğini kullanın
```

## 📈 Performans İpuçları

- **En Hızlı**: MobileNet
- **En Doğru**: VGG16
- **En Küçük**: Random Forest
- **En Dengeli**: CNN

## ❌ Sorun Giderme

### Model İndirme Sorunları

```bash
# 1. gdown paketini yükleyin
pip install gdown

# 2. Manuel indirme deneyin
python download_models.py

# 3. Hala sorun varsa, tarayıcıdan manuel indirin
# Google Drive linklerini tarayıcıda açın ve models/ klasörüne kaydedin
```

### Uygulama Çalışmıyor

```bash
# 1. Modellerin varlığını kontrol edin
ls -la models/

# 2. Python sürümünü kontrol edin (3.8+ gerekli)
python --version

# 3. Paketleri yeniden yükleyin
pip install -r requirements.txt --force-reinstall
```

### Streamlit Hatası

```bash
# Port zaten kullanımda ise
streamlit run app.py --server.port 8502

# Cache temizleme
streamlit cache clear
```

## 💾 Disk Alanı Uyarısı

⚠️ **Toplam boyut**: ~1.2 GB (5 model dosyası)
- SVM modeli en büyük dosya (532 MB)
- Yeterli disk alanınız olduğundan emin olun

## 🤝 Katkıda Bulunma

1. Fork yapın
2. Feature branch oluşturun (`git checkout -b feature/AmazingFeature`)
3. Commit yapın (`git commit -m 'Add some AmazingFeature'`)
4. Push yapın (`git push origin feature/AmazingFeature`)
5. Pull Request açın

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Geliştirici

**Deniz Altuner**
- GitHub: [@Denizaltnr](https://github.com/Denizaltnr)
- LinkedIn: [Deniz Altuner](https://linkedin.com/in/deniz-altuner-a58612180)
- Email: ben_deniz_melisa@outlook.com

## 🙏 Teşekkürler

- Dataset sağlayıcılarına
- TensorFlow ve Streamlit topluluklarına
- Açık kaynak katkıda bulunanlara

---

⭐ Bu projeyi beğendiyseniz yıldızlamayı unutmayın!

## 📝 Notlar

- Model dosyaları GitHub boyut sınırları nedeniyle repository'de yer almaz
- İlk kullanımda mutlaka `download_models.py` çalıştırın
- İnternet bağlantısı model indirme için gereklidir
- Modeller bir kez indirildikten sonra offline çalışır

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için `LICENSE` dosyasına bakın.

## 👨‍💻 Geliştirici

**Your Name**
- GitHub: [@Denizaltnr](https://github.com/Denizaltnr)
- LinkedIn: [Deniz Altuner](https://linkedin.com/in/deniz-altuner-a58612180)
- Email: ben_deniz_melisa@outlook.com

## 🙏 Teşekkürler

- Dataset sağlayıcılarına
- TensorFlow ve Streamlit topluluklarına
- Açık kaynak katkıda bulunanlara

-

---

⭐ Bu projeyi beğendiyseniz yıldızlamayı unutmayın!
