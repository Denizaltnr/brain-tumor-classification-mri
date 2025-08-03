import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import os
from pathlib import Path
import io
import cv2

# TensorFlow optional import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="🧠 Beyin Tümörü Sınıflandırması",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS stil
st.markdown("""
<style>
.main {
    padding-top: 2rem;
}
.stAlert {
    margin-bottom: 1rem;
}
.metric-container {
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    padding: 1rem;
    border-radius: 10px;
    color: white;
    text-align: center;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Model indirme fonksiyonu - DÜZELTME
@st.cache_resource
def download_models():
    """GitHub Releases'dan modelleri indir"""
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Beklenen model dosyaları (GitHub release'inizdeki isimler)
    expected_models = {
        "rf_model.pkl": "Random Forest",
        "cnn_model.h5": "CNN",
        "svm_model.pkl": "SVM", 
        "vgg16_model.h5": "VGG16",
        "mobilenet_model.h5": "MobileNet"
    }
    
    # Mevcut modelleri kontrol et
    existing_models = []
    for model_file in expected_models.keys():
        if (models_dir / model_file).exists():
            existing_models.append(model_file)
    
    if len(existing_models) >= 2:  # En az 2 model varsa devam et
        st.info(f"✅ {len(existing_models)} model zaten mevcut")
        return True
    
    try:
        with st.spinner("Model dosyaları indiriliyor... Bu işlem biraz zaman alabilir."):
            # Doğrudan release URL'si
            download_url = "https://github.com/Denizaltnr/brain-tumor-classification-mri/releases/download/v1.0.0/models.zip"
            
            st.info(f"📥 İndiriliyor: {download_url}")
            
            # ZIP dosyasını indir
            response = requests.get(download_url, stream=True, timeout=120)
            
            if response.status_code == 200:
                zip_path = "models_temp.zip"
                
                # Dosyayı kaydet
                with open(zip_path, 'wb') as f:
                    total_size = int(response.headers.get('content-length', 0))
                    downloaded = 0
                    
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                            if total_size > 0:
                                progress = downloaded / total_size
                                st.progress(progress)
                
                # ZIP'i çıkart - DÜZELTME: Doğru yere çıkart
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    # ZIP içeriğini kontrol et
                    file_list = zip_ref.namelist()
                    st.info(f"📋 ZIP içeriği: {file_list}")
                    
                    # Tüm dosyaları models/ klasörüne çıkart
                    for file_info in zip_ref.infolist():
                        # Sadece .pkl ve .h5 dosyalarını al
                        if file_info.filename.endswith(('.pkl', '.h5')):
                            # Dosya adını al (path olmadan)
                            filename = os.path.basename(file_info.filename)
                            if filename:  # Boş değilse
                                # models/ klasörüne kaydet
                                target_path = models_dir / filename
                                with zip_ref.open(file_info) as source, open(target_path, 'wb') as target:
                                    target.write(source.read())
                                st.success(f"✅ {filename} çıkartıldı")
                
                os.remove(zip_path)
                st.success("🎉 Modeller başarıyla indirildi!")
                return True
            else:
                st.error(f"❌ İndirme başarısız: HTTP {response.status_code}")
                return False
                
    except Exception as e:
        st.error(f"❌ Model indirme hatası: {str(e)}")
        st.info("""
        **Manuel İndirme:**
        1. https://github.com/Denizaltnr/brain-tumor-classification-mri/releases/tag/v1.0.0
        2. models.zip dosyasını indirin
        3. İçindeki .pkl ve .h5 dosyalarını models/ klasörüne koyun
        """)
        return False

# Model yükleme fonksiyonu - DÜZELTME
@st.cache_resource
def load_models():
    """Modelleri yükle"""
    models = {}
    
    # Önce modelleri indir
    download_success = download_models()
    
    # Model dosya yolları - DÜZELTME: Doğru dosya isimleri
    model_files = {
        'rf': 'rf_model.pkl',           # GitHub'daki gerçek isim
        'svm': 'svm_model.pkl',         # Eğer varsa
        'cnn': 'cnn_model.h5',
        'vgg16': 'vgg16_model.h5',      # Eğer varsa  
        'mobilenet': 'mobilenet_model.h5' # Eğer varsa
    }
    
    # Random Forest modelini yükle - DÜZELTME
    rf_path = Path("models") / model_files['rf']
    if rf_path.exists():
        try:
            models['rf'] = joblib.load(rf_path)
            st.success("✅ Random Forest modeli yüklendi")
        except Exception as e:
            st.error(f"❌ Random Forest yükleme hatası: {str(e)}")
            models['rf'] = None
    else:
        st.error(f"❌ Random Forest modeli bulunamadı: {rf_path}")
        models['rf'] = None
    
    # SVM modelini yükle
    svm_path = Path("models") / model_files['svm']
    if svm_path.exists():
        try:
            models['svm'] = joblib.load(svm_path)
            st.success("✅ SVM modeli yüklendi")
        except Exception as e:
            st.warning(f"⚠️ SVM yükleme hatası: {str(e)}")
            models['svm'] = None
    else:
        st.info("ℹ️ SVM modeli bulunamadı")
        models['svm'] = None
    
    # TensorFlow modelleri (eğer mevcut ise)
    if TF_AVAILABLE:
        # CNN modeli
        cnn_path = Path("models") / model_files['cnn']
        if cnn_path.exists():
            try:
                models['cnn'] = tf.keras.models.load_model(cnn_path)
                st.success("✅ CNN modeli yüklendi")
            except Exception as e:
                st.error(f"❌ CNN yükleme hatası: {str(e)}")
                models['cnn'] = None
        else:
            st.warning("⚠️ CNN modeli bulunamadı")
            models['cnn'] = None
        
        # VGG16 modeli
        vgg16_path = Path("models") / model_files['vgg16']
        if vgg16_path.exists():
            try:
                models['vgg16'] = tf.keras.models.load_model(vgg16_path)
                st.success("✅ VGG16 modeli yüklendi")
            except Exception as e:
                st.warning(f"⚠️ VGG16 yükleme hatası: {str(e)}")
                models['vgg16'] = None
        else:
            st.info("ℹ️ VGG16 modeli bulunamadı")
            models['vgg16'] = None
        
        # MobileNet modeli
        mobilenet_path = Path("models") / model_files['mobilenet']
        if mobilenet_path.exists():
            try:
                models['mobilenet'] = tf.keras.models.load_model(mobilenet_path)
                st.success("✅ MobileNet modeli yüklendi")
            except Exception as e:
                st.warning(f"⚠️ MobileNet yükleme hatası: {str(e)}")
                models['mobilenet'] = None
        else:
            st.info("ℹ️ MobileNet modeli bulunamadı")
            models['mobilenet'] = None
    else:
        st.info("ℹ️ TensorFlow mevcut değil - sadece ML modelleri aktif")
        models['cnn'] = None
        models['vgg16'] = None
        models['mobilenet'] = None
    
    return models

# Görüntü ön işleme - DÜZELTME: CNN için doğru boyut
def preprocess_image(image, model_type, target_size=(224, 224)):
    """Görüntüyü model için hazırla"""
    try:
        # PIL'den numpy array'e çevir
        img_array = np.array(image)
        
        # RGB formatına çevir (eğer RGBA ise)
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]  # Alpha kanalını kaldır
        elif len(img_array.shape) == 2:
            # Grayscale'i RGB'ye çevir
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # Model tipine göre işleme
        if model_type in ['rf', 'svm']:
            # ML modelleri için: Grayscale + flatten
            # Önce grayscale'e çevir
            if len(img_array.shape) == 3:
                img_gray = np.dot(img_array[...,:3], [0.2989, 0.5870, 0.1140])
            else:
                img_gray = img_array
            
            # 288x288'e resize et (CNN modelin beklediği boyut)
            img_resized = cv2.resize(img_gray, (288, 288))
            
            # Normalize ve flatten
            img_normalized = img_resized.astype(np.float32) / 255.0
            img_flat = img_normalized.flatten().reshape(1, -1)
            
            return img_flat
            
        elif model_type in ['cnn', 'vgg16', 'mobilenet']:
            # Deep Learning modelleri için
            # 288x288'e resize et (CNN modelin gerçek input boyutu)
            img_resized = cv2.resize(img_array, (288, 288))
            
            # RGB olduğundan emin ol
            if len(img_resized.shape) == 2:
                img_resized = np.stack([img_resized] * 3, axis=-1)
            
            # Normalize et
            img_normalized = img_resized.astype(np.float32) / 255.0
            
            # Batch dimension ekle
            img_batch = np.expand_dims(img_normalized, axis=0)
            
            return img_batch
        
        return None
        
    except Exception as e:
        st.error(f"Görüntü işleme hatası ({model_type}): {str(e)}")
        return None

# Tahmin fonksiyonu - DÜZELTME
def predict_tumor(image, models):
    """Tümör tahmini yap"""
    if models is None or not models:
        return None, None
    
    results = {}
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Random Forest tahmini
    if models.get('rf') is not None:
        try:
            processed_img = preprocess_image(image, 'rf')
            if processed_img is not None:
                rf_pred = models['rf'].predict(processed_img)[0]
                
                if hasattr(models['rf'], 'predict_proba'):
                    rf_prob = models['rf'].predict_proba(processed_img)[0]
                else:
                    # Dummy probabilities
                    rf_prob = np.zeros(4)
                    rf_prob[rf_pred] = 0.85
                    # Diğer sınıflara küçük değerler
                    for i in range(4):
                        if i != rf_pred:
                            rf_prob[i] = 0.05
                
                results['rf'] = {
                    'prediction': rf_pred,
                    'probabilities': rf_prob,
                    'max_prob': max(rf_prob)
                }
                st.info("✅ Random Forest tahmini tamamlandı")
            else:
                st.error("❌ RF için görüntü işlenemedi")
        except Exception as e:
            st.error(f"❌ Random Forest tahmin hatası: {str(e)}")
    
    # SVM tahmini
    if models.get('svm') is not None:
        try:
            processed_img = preprocess_image(image, 'svm')
            if processed_img is not None:
                svm_pred = models['svm'].predict(processed_img)[0]
                
                if hasattr(models['svm'], 'predict_proba'):
                    svm_prob = models['svm'].predict_proba(processed_img)[0]
                else:
                    svm_prob = np.zeros(4)
                    svm_prob[svm_pred] = 0.80
                    for i in range(4):
                        if i != svm_pred:
                            svm_prob[i] = 0.067
                
                results['svm'] = {
                    'prediction': svm_pred,
                    'probabilities': svm_prob,
                    'max_prob': max(svm_prob)
                }
                st.info("✅ SVM tahmini tamamlandı")
        except Exception as e:
            st.error(f"❌ SVM tahmin hatası: {str(e)}")
    
    # CNN tahmini - DÜZELTME: Doğru input boyutu
    if models.get('cnn') is not None:
        try:
            processed_img = preprocess_image(image, 'cnn')
            if processed_img is not None:
                # Input shape kontrol et
                st.info(f"🔍 CNN input shape: {processed_img.shape}")
                
                cnn_pred = models['cnn'].predict(processed_img, verbose=0)[0]
                cnn_class = np.argmax(cnn_pred)
                results['cnn'] = {
                    'prediction': cnn_class,
                    'probabilities': cnn_pred,
                    'max_prob': max(cnn_pred)
                }
                st.info("✅ CNN tahmini tamamlandı")
        except Exception as e:
            st.error(f"❌ CNN tahmin hatası: {str(e)}")
            st.error("💡 Model input boyutu ile görüntü boyutu eşleşmiyor")
    
    # VGG16 tahmini
    if models.get('vgg16') is not None:
        try:
            processed_img = preprocess_image(image, 'vgg16')
            if processed_img is not None:
                vgg_pred = models['vgg16'].predict(processed_img, verbose=0)[0]
                vgg_class = np.argmax(vgg_pred)
                results['vgg16'] = {
                    'prediction': vgg_class,
                    'probabilities': vgg_pred,
                    'max_prob': max(vgg_pred)
                }
                st.info("✅ VGG16 tahmini tamamlandı")
        except Exception as e:
            st.error(f"❌ VGG16 tahmin hatası: {str(e)}")
    
    # MobileNet tahmini  
    if models.get('mobilenet') is not None:
        try:
            processed_img = preprocess_image(image, 'mobilenet')
            if processed_img is not None:
                mobile_pred = models['mobilenet'].predict(processed_img, verbose=0)[0]
                mobile_class = np.argmax(mobile_pred)
                results['mobilenet'] = {
                    'prediction': mobile_class,
                    'probabilities': mobile_pred,
                    'max_prob': max(mobile_pred)
                }
                st.info("✅ MobileNet tahmini tamamlandı")
        except Exception as e:
            st.error(f"❌ MobileNet tahmin hatası: {str(e)}")
    
    return results, None

# Ana uygulama
def main():
    # Başlık
    st.title("🧠 Beyin Tümörü Sınıflandırması")
    st.markdown("**MRI görüntülerinden beyin tümörü türlerini tespit eden yapay zeka sistemi**")
    
    # Sınıf etiketleri
    class_names = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Sidebar
    with st.sidebar:
        st.header("📋 Proje Bilgileri")
        st.info("""
        🎯 **Hedef:** MRI görüntülerinden beyin tümörü sınıflandırması
        
        🔧 **Teknolojiler:**
        - TensorFlow/Keras
        - Scikit-learn
        - Streamlit
        - OpenCV
        
        📊 **Sınıflar:**
        - Glioma
        - Meningioma
        - Pituitary
        - No Tumor
        """)
        
        st.markdown("---")
        st.markdown("**GitHub Repository:**")
        st.markdown("[🔗 Proje Kodu](https://github.com/Denizaltnr/brain-tumor-classification-mri)")
        
        st.markdown("---")
        st.markdown("**Model Bilgileri:**")
        if TF_AVAILABLE:
            st.success("✅ TensorFlow aktif")
        else:
            st.warning("⚠️ TensorFlow mevcut değil")
        st.info("✅ Scikit-learn aktif")
        
        # Debug - Mevcut dosyaları göster
        if st.button("🔍 Dosyaları Kontrol Et"):
            models_dir = Path("models")
            if models_dir.exists():
                files = list(models_dir.glob("*"))
                if files:
                    st.write("📁 **Mevcut model dosyaları:**")
                    for file in files:
                        file_size = file.stat().st_size / (1024*1024)  # MB
                        st.write(f"- {file.name} ({file_size:.1f} MB)")
                else:
                    st.write("📁 models/ klasörü boş")
            else:
                st.write("📁 models/ klasörü yok")
    
    # Model yükleme
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            st.subheader("🤖 Model Durumu")
        with col2:
            if st.button("🔄 Modelleri Yeniden Yükle"):
                st.cache_resource.clear()
                st.rerun()
    
    models = load_models()
    
    # Ana içerik
    st.markdown("---")
    
    # Dosya yükleme
    st.subheader("📤 MRI Görüntüsü Yükleyin")
    uploaded_file = st.file_uploader(
        "JPG, JPEG veya PNG formatında MRI görüntüsü seçin",
        type=['jpg', 'jpeg', 'png'],
        help="Yüksek kaliteli MRI görüntüsü yükleyin"
    )
    
    if uploaded_file is not None:
        # Görüntüyü göster
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Yüklenen MRI Görüntüsü", use_container_width=True)
            
            # Görüntü bilgileri
            st.caption(f"📏 Boyut: {image.size[0]} x {image.size[1]} piksel")
            st.caption(f"📁 Format: {image.format}")
            st.caption(f"🎨 Mod: {image.mode}")
        
        with col2:
            if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
                if models is None or not any(models.values()):
                    st.error("❌ Hiçbir model yüklenemedi. Lütfen modelleri kontrol edin.")
                else:
                    with st.spinner("Analiz ediliyor..."):
                        results, _ = predict_tumor(image, models)
                        
                        if results:
                            st.subheader("📊 Analiz Sonuçları")
                            
                            # Her model için sonuçları göster
                            model_names = {
                                'rf': '🌳 Random Forest',
                                'svm': '🔷 SVM', 
                                'cnn': '🧠 CNN',
                                'vgg16': '🏗️ VGG16',
                                'mobilenet': '📱 MobileNet'
                            }
                            
                            for model_key, model_name in model_names.items():
                                if model_key in results:
                                    pred = results[model_key]['prediction']
                                    prob = results[model_key]['probabilities']
                                    confidence = results[model_key]['max_prob']
                                    
                                    st.markdown(f"**{model_name}:**")
                                    st.success(f"**Tahmin:** {class_names[pred]}")
                                    st.info(f"**Güven:** {confidence:.2%}")
                                    
                                    # Grafik
                                    fig, ax = plt.subplots(figsize=(8, 3))
                                    bars = ax.bar(class_names, prob, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                                    ax.set_ylabel('Olasılık')
                                    ax.set_title(f'{model_name} - Sınıf Olasılıkları')
                                    ax.set_ylim(0, 1)
                                    
                                    # En yüksek değeri vurgula
                                    max_idx = np.argmax(prob)
                                    bars[max_idx].set_color('#FF4444')
                                    
                                    plt.xticks(rotation=45)
                                    plt.tight_layout()
                                    st.pyplot(fig)
                                    plt.close()
                                    
                                    st.markdown("---")
                            
                            # Genel değerlendirme
                            st.subheader("🎯 Genel Değerlendirme")
                            
                            # En güvenilir modelin sonucunu al
                            best_result = None
                            best_confidence = 0
                            
                            for model_key, result in results.items():
                                if result['max_prob'] > best_confidence:
                                    best_confidence = result['max_prob']
                                    best_result = result
                                    best_model = model_names.get(model_key, model_key)
                            
                            if best_result:
                                if best_confidence > 0.8:
                                    st.success(f"🟢 {best_model} ile yüksek güven seviyesi: {best_confidence:.1%}")
                                elif best_confidence > 0.6:
                                    st.warning(f"🟡 {best_model} ile orta güven seviyesi: {best_confidence:.1%}")
                                else:
                                    st.error(f"🔴 Düşük güven seviyesi: {best_confidence:.1%}")
                            
                            # Uyarı metni
                            st.warning("⚠️ **Önemli:** Bu sistem sadece araştırma amaçlıdır. Kesin teşhis için mutlaka uzman doktor görüşü alınmalıdır.")
                        
                        else:
                            st.error("❌ Tahmin yapılamadı. Lütfen farklı bir görüntü deneyin.")
    
    # Demo açıklaması
    st.markdown("---")
    st.subheader("ℹ️ Nasıl Kullanılır?")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1️⃣ Görüntü Yükle**
        - MRI beyin görüntüsü seçin
        - JPG, JPEG veya PNG formatı
        - Net ve kaliteli görüntü tercih edin
        """)
    
    with col2:
        st.markdown("""
        **2️⃣ Analiz Et**
        - "Analiz Et" butonuna tıklayın
        - AI modelleri görüntüyü işler
        - Sonuçlar otomatik görüntülenir
        """)
    
    with col3:
        st.markdown("""
        **3️⃣ Sonuçları Değerlendir**
        - Tahmin sonuçlarını inceleyin
        - Güven skorlarına dikkat edin
        - Uzman görüşü almayı unutmayın
        """)

if __name__ == "__main__":
    main()
