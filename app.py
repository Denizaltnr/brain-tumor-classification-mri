import streamlit as st
import numpy as np
import pandas as pd
from PIL import Image
import joblib
import cv2
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
import os
from pathlib import Path
import io

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

# Model indirme fonksiyonu
@st.cache_resource
def download_models():
    """GitHub Releases'dan modelleri indir"""
    models_dir = Path("models")
    
    # Modeller zaten varsa skip et
    if models_dir.exists() and any(models_dir.iterdir()):
        return True
    
    models_dir.mkdir(exist_ok=True)
    
    try:
        with st.spinner("Model dosyaları indiriliyor... Bu işlem biraz zaman alabilir."):
            # GitHub Releases API
            api_url = "https://api.github.com/repos/Denizaltnr/brain-tumor-classification-mri/releases/latest"
            response = requests.get(api_url, timeout=30)
            
            if response.status_code != 200:
                return False
                
            release_data = response.json()
            
            for asset in release_data.get('assets', []):
                if asset['name'].endswith('.zip'):
                    download_url = asset['browser_download_url']
                    
                    # Dosyayı indir
                    file_response = requests.get(download_url, stream=True, timeout=60)
                    zip_path = f"temp_{asset['name']}"
                    
                    with open(zip_path, 'wb') as f:
                        for chunk in file_response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Zip'i çıkar
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall('models')
                    
                    os.remove(zip_path)
                    return True
                    
        return False
        
    except Exception as e:
        st.error(f"Model indirme hatası: {str(e)}")
        return False

# Model yükleme fonksiyonu
@st.cache_resource
def load_models():
    """Modelleri yükle"""
    models = {}
    
    # Önce modelleri indir
    download_success = download_models()
    
    if not download_success:
        st.warning("⚠️ Modeller indirilemedi. Demo modunda çalışılıyor.")
        return None
    
    try:
        # Random Forest modelini yükle
        rf_path = Path("models/random_forest_model.pkl")
        if rf_path.exists():
            models['rf'] = joblib.load(rf_path)
            st.success("✅ Random Forest modeli yüklendi")
        else:
            st.error("❌ Random Forest modeli bulunamadı")
            
    except Exception as e:
        st.error(f"❌ Random Forest modeli yüklenemedi: {str(e)}")
        models['rf'] = None
    
    # TensorFlow modeli (optional)
    if TF_AVAILABLE:
        try:
            cnn_path = Path("models/cnn_model.h5")
            if cnn_path.exists():
                models['cnn'] = tf.keras.models.load_model(cnn_path)
                st.success("✅ CNN modeli yüklendi")
            else:
                st.warning("⚠️ CNN modeli bulunamadı")
                models['cnn'] = None
        except Exception as e:
            st.warning(f"⚠️ CNN modeli yüklenemedi: {str(e)}")
            models['cnn'] = None
    else:
        st.info("ℹ️ TensorFlow mevcut değil - sadece Random Forest aktif")
        models['cnn'] = None
    
    return models

# Görüntü ön işleme
def preprocess_image(image, target_size=(224, 224)):
    """Görüntüyü model için hazırla"""
    try:
        # PIL'den numpy array'e çevir
        img_array = np.array(image)
        
        # RGB'ye çevir (eğer RGBA ise)
        if img_array.shape[-1] == 4:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        elif len(img_array.shape) == 3 and img_array.shape[-1] == 3:
            img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        # Yeniden boyutlandır
        img_resized = cv2.resize(img_array, target_size)
        
        # Normalize et (0-1 arası)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
        
    except Exception as e:
        st.error(f"Görüntü işleme hatası: {str(e)}")
        return None

# Tahmin fonksiyonu
def predict_tumor(image, models):
    """Tümör tahmini yap"""
    if models is None:
        return None, None
        
    processed_img = preprocess_image(image)
    if processed_img is None:
        return None, None
    
    results = {}
    
    # Random Forest tahmini
    if models.get('rf') is not None:
        try:
            # Görüntüyü flatten et
            img_flat = processed_img.flatten().reshape(1, -1)
            rf_pred = models['rf'].predict(img_flat)[0]
            rf_prob = models['rf'].predict_proba(img_flat)[0]
            results['rf'] = {
                'prediction': rf_pred,
                'probabilities': rf_prob,
                'max_prob': max(rf_prob)
            }
        except Exception as e:
            st.error(f"Random Forest tahmin hatası: {str(e)}")
    
    # CNN tahmini
    if models.get('cnn') is not None:
        try:
            img_batch = np.expand_dims(processed_img, axis=0)
            cnn_pred = models['cnn'].predict(img_batch)[0]
            cnn_class = np.argmax(cnn_pred)
            results['cnn'] = {
                'prediction': cnn_class,
                'probabilities': cnn_pred,
                'max_prob': max(cnn_pred)
            }
        except Exception as e:
            st.error(f"CNN tahmin hatası: {str(e)}")
    
    return results, processed_img

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
            st.image(image, caption="Yüklenen MRI Görüntüsü", use_column_width=True)
            
            # Görüntü bilgileri
            st.caption(f"📏 Boyut: {image.size[0]} x {image.size[1]} piksel")
            st.caption(f"📁 Format: {image.format}")
            st.caption(f"🎨 Mod: {image.mode}")
        
        with col2:
            if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
                if models is None:
                    st.error("❌ Model yüklenemedi. Lütfen modelleri kontrol edin.")
                else:
                    with st.spinner("Analiz ediliyor..."):
                        results, processed_img = predict_tumor(image, models)
                        
                        if results:
                            st.subheader("📊 Analiz Sonuçları")
                            
                            # Random Forest sonuçları
                            if 'rf' in results:
                                rf_pred = results['rf']['prediction']
                                rf_prob = results['rf']['probabilities']
                                rf_confidence = results['rf']['max_prob']
                                
                                st.markdown("**🌳 Random Forest Modeli:**")
                                st.success(f"**Tahmin:** {class_names[rf_pred]}")
                                st.info(f"**Güven Skoru:** {rf_confidence:.2%}")
                                
                                # Olasılık grafiği
                                fig, ax = plt.subplots(figsize=(8, 4))
                                bars = ax.bar(class_names, rf_prob, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                                ax.set_ylabel('Olasılık')
                                ax.set_title('Random Forest - Sınıf Olasılıkları')
                                ax.set_ylim(0, 1)
                                
                                # En yüksek değeri vurgula
                                max_idx = np.argmax(rf_prob)
                                bars[max_idx].set_color('#FF4444')
                                
                                # Değerleri bar üzerine yaz
                                for i, v in enumerate(rf_prob):
                                    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                                
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            
                            # CNN sonuçları (eğer mevcut ise)
                            if 'cnn' in results:
                                cnn_pred = results['cnn']['prediction']
                                cnn_prob = results['cnn']['probabilities']
                                cnn_confidence = results['cnn']['max_prob']
                                
                                st.markdown("**🧠 CNN Modeli:**")
                                st.success(f"**Tahmin:** {class_names[cnn_pred]}")
                                st.info(f"**Güven Skoru:** {cnn_confidence:.2%}")
                                
                                # CNN olasılık grafiği
                                fig, ax = plt.subplots(figsize=(8, 4))
                                bars = ax.bar(class_names, cnn_prob, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4'])
                                ax.set_ylabel('Olasılık')
                                ax.set_title('CNN - Sınıf Olasılıkları')
                                ax.set_ylim(0, 1)
                                
                                # En yüksek değeri vurgula
                                max_idx = np.argmax(cnn_prob)
                                bars[max_idx].set_color('#FF4444')
                                
                                # Değerleri bar üzerine yaz
                                for i, v in enumerate(cnn_prob):
                                    ax.text(i, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
                                
                                plt.xticks(rotation=45)
                                plt.tight_layout()
                                st.pyplot(fig)
                                plt.close()
                            
                            # Genel değerlendirme
                            st.markdown("---")
                            st.subheader("🎯 Genel Değerlendirme")
                            
                            if 'rf' in results:
                                if results['rf']['max_prob'] > 0.8:
                                    st.success("🟢 Yüksek güven seviyesi ile tahmin yapıldı")
                                elif results['rf']['max_prob'] > 0.6:
                                    st.warning("🟡 Orta güven seviyesi - ek inceleme önerilir")
                                else:
                                    st.error("🔴 Düşük güven seviyesi - uzman görüşü alınmalı")
                            
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
