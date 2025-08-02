import os
import cv2
import numpy as np
import joblib
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score

# Dosya yolları
data_dir = r"C:\Users\ben_d\Desktop\VeriProje\Brain-Tumor-Classification-MRI"
test_dir = os.path.join(data_dir, "Testing")
model_path = os.path.join(data_dir, "svm_model.pkl")

# Modeli yükle
model = joblib.load(model_path)
print("SVM modeli yüklendi.")

# Kategori etiketleri
categories = sorted(os.listdir(test_dir))
print("Kategoriler:", categories)

IMG_SIZE = 224  # Görüntü boyutu

# Ön işleme fonksiyonu (flatten vektör)
def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)  # Gri tonlama (tek kanal)
    img = img / 255.0
    return img.flatten()

# Tahmin ve değerlendirme
results = []
y_true = []
y_pred = []

for category in categories:
    category_path = os.path.join(test_dir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        feature_vector = preprocess_image(img_path)
        prediction = model.predict([feature_vector])[0]
        results.append((img_path, category, prediction))
        y_true.append(category)
        y_pred.append(prediction)

# Doğruluk
accuracy = accuracy_score(y_true, y_pred)
print(f"\nSVM Modeli Test Doğruluk Oranı: {accuracy * 100:.2f}%")

# Görsel gösterim
batch_size = 16
total_images = len(results)
num_batches = (total_images + batch_size - 1) // batch_size

for batch_idx in range(num_batches):
    plt.figure(figsize=(12, 12))
    batch_results = results[batch_idx * batch_size: (batch_idx + 1) * batch_size]
    for i, (img_path, true_label, pred_label) in enumerate(batch_results):
        img_show = cv2.imread(img_path)
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 4, i + 1)
        plt.imshow(img_show)
        plt.axis('off')
        plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label}", fontsize=8)
    plt.tight_layout(pad=3.0)
    plt.show()
