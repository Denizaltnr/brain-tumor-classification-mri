import streamlit as st
import os
import numpy as np
from PIL import Image
import joblib
from tensorflow.keras.models import load_model
from utils.preprocessing import preprocess_image, flatten_for_classic_models
import streamlit as st
import requests
import zipfile
import os
from pathlib import Path

@st.cache_resource
def download_models():
    """GitHub Releases'dan modelleri indir"""
    models_dir = Path("models")
    
    if models_dir.exists() and any(models_dir.iterdir()):
        return  # Modeller zaten var
    
    models_dir.mkdir(exist_ok=True)
    
    with st.spinner("Model dosyaları indiriliyor..."):
        try:
            # GitHub Releases API
            api_url = "https://api.github.com/repos/Denizaltnr/brain-tumor-classification-mri/releases/latest"
            response = requests.get(api_url)
            release_data = response.json()
            
            for asset in release_data['assets']:
                if asset['name'].endswith('.zip'):
                    download_url = asset['browser_download_url']
                    
                    # Dosyayı indir
                    file_response = requests.get(download_url, stream=True)
                    zip_path = f"temp_{asset['name']}"
                    
                    with open(zip_path, 'wb') as f:
                        for chunk in file_response.iter_content(chunk_size=8192):
                            f.write(chunk)
                    
                    # Zip'i çıkar
                    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                        zip_ref.extractall('models')
                    
                    os.remove(zip_path)
                    st.success("Modeller başarıyla yüklendi!")
                    return
                    
        except Exception as e:
            st.error(f"Model yükleme hatası: {e}")
            st.info("Lütfen manuel olarak model dosyalarını yükleyin.")

# Ana uygulamanın başında çağır
download_models()

# 🔍 Model yolları
MODEL_PATHS = {
    "CNN": "models/cnn_model.h5",
    "VGG16": "models/vgg16_model.h5",  # Düzeltildi: çift underscore kaldırıldı
    "MobileNet": "models/mobilenet_model.h5",
    "SVM": "models/svm_model.pkl",
    "Random Forest": "models/rf_model.pkl"
}

# 🧠 Sınıf etiketleri
class_labels = {
    0: "glioma_tumor",
    1: "meningioma_tumor",
    2: "no_tumor",
    3: "pituitary_tumor"
}

st.set_page_config(page_title="Beyin Tümörü Sınıflandırması", layout="centered")
st.title("🧠 Beyin Tümörü Sınıflandırma Uygulaması")
st.write("MRI görüntüsüne göre tümör türünü tahmin eder.")

uploaded_image = st.file_uploader("🔍 MRI Görselini Yükleyin", type=["jpg", "jpeg", "png"])
model_choice = st.selectbox("📌 Model Seçin", list(MODEL_PATHS.keys()))

if uploaded_image and model_choice:
    image = Image.open(uploaded_image)
    st.image(image, caption="Yüklenen Görsel", use_column_width=True)

    try:
        # 📦 Model dosyasının varlığını kontrol et
        model_path = MODEL_PATHS[model_choice]
        if not os.path.exists(model_path):
            st.error(f"⚠️ Model dosyası bulunamadı: {model_path}")
            st.stop()

        # 🔧 Görseli işle - TÜM MODELLER İÇİN AYNI BOYUT
        image_array = preprocess_image(image, model_type=model_choice)

        # 📦 Modeli yükle ve tahmin yap
        if model_choice in ["CNN", "VGG16", "MobileNet"]:
            model = load_model(model_path)
            
            # CNN için dinamik boyut kontrolü (inference kodundan)
            if model_choice == "CNN":
                expected_shape = model.input_shape[1:3]  # (height, width)
                current_shape = image_array.shape[1:3]
                
                if expected_shape != current_shape:
                    st.info(f"🔄 CNN boyut uyumsuzluğu: {current_shape} -> {expected_shape}. Yeniden boyutlandırılıyor...")
                    # Yeniden boyutlandır
                    from PIL import Image as PILImage
                    img_resized = PILImage.fromarray((image_array[0] * 255).astype(np.uint8))
                    img_resized = img_resized.resize(expected_shape)
                    image_array = np.array(img_resized) / 255.0
                    image_array = np.expand_dims(image_array, axis=0)
            
            prediction = model.predict(image_array)
            predicted_class = int(np.argmax(prediction))

        elif model_choice in ["SVM", "Random Forest"]:
            model = joblib.load(model_path)
            
            # Model tipine göre özel flatten
            if model_choice == "SVM":
                # SVM: Grayscale flatten (50,176 features)
                image_array = flatten_for_classic_models(image_array, "SVM")
            elif model_choice == "Random Forest":
                # RF: MobileNet feature extraction
                image_array = flatten_for_classic_models(image_array, "Random Forest")
                
            predicted_class = model.predict(image_array)[0]

            # Klasik modeller string döndürüyor olabilir, sayıya çevir
            if isinstance(predicted_class, str):
                predicted_class = int(list(class_labels.keys())[list(class_labels.values()).index(predicted_class)])

        class_name = class_labels.get(predicted_class, "Bilinmeyen Sınıf")
        st.success(f"✅ Tahmin: {class_name}")

    except Exception as e:
        st.error(f"⚠️ Tahmin sırasında hata oluştu: {e}")
        st.info("💡 Model tipine göre preprocessing:")
        st.info("• CNN: RGB, dinamik boyut")  
        st.info("• SVM: GRAYSCALE, 224x224 (50,176 features)")
        st.info("• RF: RGB + MobileNet features")
        st.info("• VGG16/MobileNet: RGB, 224x224")
