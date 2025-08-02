from PIL import Image
import numpy as np

def preprocess_image(image, model_type="CNN"):
    """
    MRI görselini verilen model tipine göre uygun biçime getirir.
    """
    # Model tipine göre farklı preprocessing
    if model_type == "CNN":
        # CNN: RGB + 224x224
        image = image.convert("RGB")
        target_size = (224, 224)
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
    elif model_type == "SVM":
        # SVM: GRAYSCALE + 224x224 (inference kodundan öğrenildi)
        image = image.convert("L")  # Grayscale'e çevir
        target_size = (224, 224)
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        # SVM için channel dimension ekleme
        image_array = np.expand_dims(image_array, axis=-1)  # (1, 224, 224, 1)
        
    elif model_type == "Random Forest":
        # RF: RGB + 224x224 (MobileNet feature extraction için)
        image = image.convert("RGB")
        target_size = (224, 224)
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        
    else:
        # VGG16, MobileNet: Standard RGB + 224x224
        image = image.convert("RGB")
        target_size = (224, 224)
        image = image.resize(target_size)
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

    return image_array


def extract_features_for_rf(image_array):
    """
    Random Forest için MobileNet ile feature extraction.
    Inference kodundan alınmıştır.
    """
    try:
        from tensorflow.keras.applications import MobileNet
        from tensorflow.keras.applications.mobilenet import preprocess_input
        
        # MobileNet modelini yükle (sadece feature extraction için)
        base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))
        
        # MobileNet preprocessing
        img_preprocessed = preprocess_input(image_array * 255.0)  # 0-1 den 0-255'e geri çevir
        
        # Feature extraction
        features = base_model.predict(img_preprocessed, verbose=0)
        return features.flatten()
        
    except ImportError:
        # Eğer MobileNet import edilemezse, raw pixel kullan
        return image_array.flatten()


def flatten_for_classic_models(image_array, model_type="SVM"):
    """
    Klasik modeller için preprocessing.
    """
    if model_type == "SVM":
        # SVM grayscale flatten: (1, 224, 224, 1) -> (1, 50176)
        return image_array.reshape((image_array.shape[0], -1))
    
    elif model_type == "Random Forest":
        # RF için MobileNet features kullan
        features = extract_features_for_rf(image_array)
        return features.reshape(1, -1)  # (1, feature_count) formatına getir
    
    else:
        # Diğer modeller için standart flatten
        return image_array.reshape((image_array.shape[0], -1))


def get_optimal_size_for_features(target_features, channels=3):
    """
    Hedef feature sayısı için optimal görüntü boyutunu hesaplar.
    """
    pixels_needed = target_features // channels
    
    # Mükemmel kare kök varsa onu kullan
    sqrt_val = int(np.sqrt(pixels_needed))
    if sqrt_val * sqrt_val * channels == target_features:
        return (sqrt_val, sqrt_val)
    
    # Değilse, en yakın faktörleri bul
    for w in range(sqrt_val - 10, sqrt_val + 10):
        if pixels_needed % w == 0:
            h = pixels_needed // w
            if abs(w - h) < 20:  # Boyutlar çok farklı olmasın
                return (w, h)
    
    # Son çare: yaklaşık değer
    return (sqrt_val, (pixels_needed // sqrt_val) + 1)


# Tüm modeller 224x224 boyutunda eğitildiği için bu fonksiyonlar artık gerekli değil
# Ancak geriye dönük uyumluluk için bırakılabilir

def get_model_input_info():
    """
    Tüm modeller için input bilgilerini döndürür.
    """
    return {
        "input_shape": (224, 224, 3),
        "total_features": 224 * 224 * 3,  # 150,528
        "preprocessing": "RGB normalization (0-1)"
    }


def adaptive_preprocess_image(image, model_type="CNN", expected_features=None):
    """
    Gelişmiş preprocessing fonksiyonu - tüm modeller 224x224 boyutunda eğitilmiş.
    """
    # Tüm modeller aynı boyutta olduğu için standart preprocessing yeterli
    return preprocess_image(image, model_type)