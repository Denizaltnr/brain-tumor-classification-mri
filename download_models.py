import gdown
import os

model_links = {
    "cnn_model.h5": "https://drive.google.com/uc?id=1fSVLr3PjR7YsUpBKgU4MvqxG9YzvuN6k",
    "svm_model.pkl": "https://drive.google.com/uc?id=1PUysW4CWS69HnAOTZTkENOhdi56Bm7dt",
    "rf_model.pkl": "https://drive.google.com/uc?id=1sJq3lUssRlZrxUK6fFkj9U4WhY9z1BRp",
    "vgg16_model.h5": "https://drive.google.com/uc?id=18aoVfmxPaLGV902UulBhQKNrqHEt1TVt",
    "mobilenet_model.h5": "https://drive.google.com/uc?id=1CDPS2wtC8BTFeCWQQn5htIi9lms4ROk9",
}

os.makedirs("models", exist_ok=True)

for filename, url in model_links.items():
    output_path = os.path.join("models", filename)
    if not os.path.exists(output_path):
        print(f"Downloading {filename}...")
        gdown.download(url, output_path, quiet=False)
