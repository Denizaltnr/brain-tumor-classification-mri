import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from sklearn.metrics import accuracy_score

# Ayarlar
IMG_SIZE = 224
data_dir = r"C:\Users\ben_d\Desktop\VeriProje\Brain-Tumor-Classification-MRI"
test_dir = os.path.join(data_dir, "Testing")

categories = sorted(os.listdir(test_dir))

model = load_model("mobilenet_brain_tumor_model.h5")

def preprocess_image(img_path):
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)
    return img

results = []
y_true = []
y_pred = []

for category in categories:
    category_path = os.path.join(test_dir, category)
    for img_name in os.listdir(category_path):
        img_path = os.path.join(category_path, img_name)
        img_preprocessed = preprocess_image(img_path)
        preds = model.predict(img_preprocessed)
        pred_label = categories[np.argmax(preds)]
        confidence = np.max(preds)
        results.append((img_path, category, pred_label, confidence))
        y_true.append(category)
        y_pred.append(pred_label)

accuracy = accuracy_score(y_true, y_pred)
print(f"Test Doğruluk Oranı: {accuracy * 100:.2f}%")

# 16'şarlı gruplara bölüp gösterelim
batch_size = 16
total_images = len(results)
num_batches = (total_images + batch_size - 1) // batch_size  # yukarı yuvarlama

for batch_idx in range(num_batches):
    plt.figure(figsize=(12, 12))
    batch_results = results[batch_idx*batch_size : (batch_idx+1)*batch_size]
    for i, (img_path, true_label, pred_label, confidence) in enumerate(batch_results):
        img_show = cv2.imread(img_path)
        img_show = cv2.cvtColor(img_show, cv2.COLOR_BGR2RGB)
        plt.subplot(4, 4, i+1)
        plt.imshow(img_show)
        plt.axis('off')
        plt.title(f"Gerçek: {true_label}\nTahmin: {pred_label}\nGüven: {confidence:.2f}", fontsize=8)
    plt.tight_layout(pad=3.0)
    plt.show()
