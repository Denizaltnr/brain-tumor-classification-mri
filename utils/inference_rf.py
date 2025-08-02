import os
import numpy as np
import cv2
import joblib
import matplotlib.pyplot as plt
from keras.applications import MobileNet
from keras.applications.mobilenet import preprocess_input
from keras.preprocessing.image import img_to_array

# 🔹 1. Ayarlar
IMG_SIZE = (224, 224)
data_dir = r"C:\Users\ben_d\Desktop\VeriProje\Brain-Tumor-Classification-MRI"
test_dir = os.path.join(data_dir, "Testing")
model_path = os.path.join(data_dir, "random_forest_model.pkl")

# 🔹 2. Sınıf etiketlerini al
categories = sorted(os.listdir(test_dir))

# 🔹 3. Özellik çıkarımı için MobileNet (üst katmanlar olmadan)
base_model = MobileNet(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# 🔹 4. Görüntüden öznitelik çıkarma
def extract_features(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, IMG_SIZE)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    features = base_model.predict(img, verbose=0)
    return features.flatten()

# 🔹 5. Modeli yükle
if not os.path.exists(model_path):
    raise FileNotFoundError(f"Model dosyası bulunamadı: {model_path}")
rf_model = joblib.load(model_path)

# 🔹 6. Tahmin ve sonuçları saklama
results = []
y_true = []
y_pred = []

for category in categories:
    category_path = os.path.join(test_dir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        try:
            features = extract_features(img_path)
            prediction = rf_model.predict([features])[0]
            confidence = np.max(rf_model.predict_proba([features]))
            results.append((img_path, category, prediction, confidence))
            y_true.append(category)
            y_pred.append(prediction)
        except Exception as e:
            print(f"Hata: {img_path} => {e}")

# 🔹 7. Doğruluk oranı
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_true, y_pred)
print(f"\n🎯 Test Doğruluk Oranı: {accuracy * 100:.2f}%")

# 🔹 8. Görselleştirme (16'şarlık gruplar)
batch_size = 16
total_images = len(results)
num_batches = (total_images + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    plt.figure(figsize=(12, 12))
    batch_results = results[batch_idx * batch_size : (batch_idx + 1) * batch_size]
    for i, (img_path, true_label, pred_label, confidence) in enumerate(batch_results):
        img_show = cv2.imread(img_path)
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img_show)
        plt.axis('off')
        plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label}\nGüven: {confidence:.2f}", fontsize=8)
    plt.tight_layout(pad=3.0)
    plt.show()
