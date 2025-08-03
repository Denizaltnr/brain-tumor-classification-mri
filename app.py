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

# TensorFlow optional import
try:
    import tensorflow as tf
    TF_AVAILABLE = True
    st.success("✅ TensorFlow mevcut")
except ImportError:
    TF_AVAILABLE = False
    st.warning("⚠️ TensorFlow mevcut değil")

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

# Model indirme fonksiyonu - Düzeltildi
@st.cache_resource
def download_models():
    """GitHub Releases'dan modelleri indir"""
    models_dir = Path("models")
    
    # Modeller zaten varsa skip et
    if models_dir.exists() and any(models_dir.iterdir()):
        st.info("✅ Modeller zaten mevcut")
        return True
    
    models_dir.mkdir(exist_ok=True)
    
    try:
        with st.spinner("Model dosyaları indiriliyor... Bu işlem biraz zaman alabilir."):
            # Doğrudan GitHub Releases URL'si
            repo_url = "https://github.com/Denizaltnr/brain-tumor-classification-mri"
            
            # Alternatif indirme yöntemleri
            download_urls = [
                f"{repo_url}/releases/download/v1.0.0/models.zip",
                f"{repo_url}/releases/download/v1.0/models.zip",
                f"{repo_url}/releases/latest/download/models.zip"
            ]
            
            for download_url in download_urls:
                try:
                    st.info(f"🔄 Deneniyor: {download_url}")
                    
                    # Dosyayı indir
                    response = requests.get(download_url, stream=True, timeout=60)
                    
                    if response.status_code == 200:
                        zip_path = "models_temp.zip"
                        
                        with open(zip_path, 'wb') as f:
                            for chunk in response.iter_content(chunk_size=8192):
                                if chunk:
                                    f.write(chunk)
                        
                        # Zip'i çıkar
                        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                            zip_ref.extractall('.')
                        
                        os.remove(zip_path)
                        st.success("✅ Modeller başarıyla indirildi!")
                        return True
                        
                except Exception as e:
                    st.warning(f"⚠️ Bu URL başarısız: {str(e)}")
                    continue
            
            # Hiçbiri çalışmazsa manuel indirme talimatı
            st.error("❌ Otomatik indirme başarısız!")
            st.info("""
            **Manuel İndirme Adımları:**
            1. Bu linki ziyaret edin: https://github.com/Denizaltnr/brain-tumor-classification-mri/releases
            2. En son release'den models.zip dosyasını indirin
            3. Proje dizininizde çıkartın
            4. models/ klasörünün oluştuğundan emin olun
            """)
            return False
                    
    except Exception as e:
        st.error(f"Model indirme hatası: {str(e)}")
        return False

# Model yükleme fonksiyonu - Düzeltildi
@st.cache_resource
def load_models():
    """Modelleri yükle"""
    models = {}
    
    # Önce modelleri indir
    download_success = download_models()
    
    if not download_success:
        st.warning("⚠️ Modeller indirilemedi. Mevcut modeller kontrol ediliyor...")
    
    # Model dosya yolları - Düzeltildi
    model_paths = {
        'rf': [
            Path("models/rf_model.pkl"),
            Path("models/random_forest_model.pkl"),
            Path("models/rf.pkl")
        ],
        'cnn': [
            Path("models/cnn_model.h5"),
            Path("models/cnn.h5"),
            Path("models/brain_tumor_cnn.h5")
        ]
    }
    
    # Random Forest modelini yükle
    rf_loaded = False
    for rf_path in model_paths['rf']:
        if rf_path.exists():
            try:
                models['rf'] = joblib.load(rf_path)
                st.success(f"✅ Random Forest modeli yüklendi: {rf_path.name}")
                rf_loaded = True
                break
            except Exception as e:
                st.warning(f"⚠️ {rf_path.name} yüklenemedi: {str(e)}")
                continue
    
    if not rf_loaded:
        st.error("❌ Random Forest modeli bulunamadı")
        models['rf'] = None
    
    # TensorFlow modeli (optional)
    if TF_AVAILABLE:
        cnn_loaded = False
        for cnn_path in model_paths['cnn']:
            if cnn_path.exists():
                try:
                    models['cnn'] = tf.keras.models.load_model(cnn_path)
                    st.success(f"✅ CNN modeli yüklendi: {cnn_path.name}")
                    cnn_loaded = True
                    break
                except Exception as e:
                    st.warning(f"⚠️ {cnn_path.name} yüklenemedi: {str(e)}")
                    continue
        
        if not cnn_loaded:
            st.warning("⚠️ CNN modeli bulunamadı")
            models['cnn'] = None
    else:
        st.info("ℹ️ TensorFlow mevcut değil - sadece Random Forest aktif")
        models['cnn'] = None
    
    return models

# Görüntü ön işleme - Düzeltildi
def preprocess_image(image, target_size=(224, 224)):
    """Görüntüyü model için hazırla"""
    try:
        # PIL'den numpy array'e çevir
        img_array = np.array(image)
        
        # RGB formatına çevir (eğer RGBA ise)
        if len(img_array.shape) == 3 and img_array.shape[-1] == 4:
            img_array = img_array[:, :, :3]  # Alpha kanalını kaldır
        
        # Eğer grayscale ise RGB'ye çevir
        if len(img_array.shape) == 2:
            img_array = np.stack([img_array] * 3, axis=-1)
        
        # PIL ile yeniden boyutlandır
        image_resized = image.resize(target_size, Image.Resampling.LANCZOS)
        img_resized = np.array(image_resized)
        
        # RGB'ye çevir (tekrar kontrol)
        if len(img_resized.shape) == 3 and img_resized.shape[-1] == 4:
            img_resized = img_resized[:, :, :3]
        elif len(img_resized.shape) == 2:
            img_resized = np.stack([img_resized] * 3, axis=-1)
        
        # Normalize et (0-1 arası)
        img_normalized = img_resized.astype(np.float32) / 255.0
        
        return img_normalized
        
    except Exception as e:
        st.error(f"Görüntü işleme hatası: {str(e)}")
        return None

# Tahmin fonksiyonu - Düzeltildi
def predict_tumor(image, models):
    """Tümör tahmini yap"""
    if models is None:
        st.error("❌ Modeller yüklenemedi")
        return None, None
        
    processed_img = preprocess_image(image)
    if processed_img is None:
        st.error("❌ Görüntü işlenemedi")
        return None, None
    
    results = {}
    
    # Random Forest tahmini
    if models.get('rf') is not None:
        try:
            # Görüntüyü flatten et
            img_flat = processed_img.flatten().reshape(1, -1)
            rf_pred = models['rf'].predict(img_flat)[0]
            
            # Predict_proba varsa kullan
            if hasattr(models['rf'], 'predict_proba'):
                rf_prob = models['rf'].predict_proba(img_flat)[0]
            else:
                # Eğer predict_proba yoksa dummy probabilities oluştur
                rf_prob = np.zeros(4)
                rf_prob[rf_pred] = 0.85
                rf_prob = rf_prob / rf_prob.sum()
            
            results['rf'] = {
                'prediction': rf_pred,
                'probabilities': rf_prob,
                'max_prob': max(rf_prob)
            }
            st.info("✅ Random Forest tahmini tamamlandı")
            
        except Exception as e:
            st.error(f"❌ Random Forest tahmin hatası: {str(e)}")
    else:
        st.error("❌ Random Forest modeli mevcut değil")
    
    # CNN tahmini
    if models.get('cnn') is not None:
        try:
            img_batch = np.expand_dims(processed_img, axis=0)
            cnn_pred = models['cnn'].predict(img_batch, verbose=0)[0]
            cnn_class = np.argmax(cnn_pred)
            results['cnn'] = {
                'prediction': cnn_class,
                'probabilities': cnn_pred,
                'max_prob': max(cnn_pred)
            }
            st.info("✅ CNN tahmini tamamlandı")
            
        except Exception as e:
            st.error(f"❌ CNN tahmin hatası: {str(e)}")
    
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
        
        # Debug bilgileri
        if st.button("🔍 Debug Bilgileri"):
            st.write("**Mevcut dosyalar:**")
            models_dir = Path("models")
            if models_dir.exists():
                for file in models_dir.iterdir():
                    st.write(f"- {file.name}")
            else:
                st.write("models/ klasörü yok")
    
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
            # Düzeltme: use_column_width -> use_container_width
            st.image(image, caption="Yüklenen MRI Görüntüsü", use_container_width=True)
            
            # Görüntü bilgileri
            st.caption(f"📏 Boyut: {image.size[0]} x {image.size[1]} piksel")
            st.caption(f"📁 Format: {image.format}")
            st.caption(f"🎨 Mod: {image.mode}")
        
        with col2:
            if st.button("🔍 Analiz Et", type="primary", use_container_width=True):
                if models is None or (models.get('rf') is None and models.get('cnn') is None):
                    st.error("❌ Hiçbir model yüklenemedi. Lütfen modelleri kontrol edin.")
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
