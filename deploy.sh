#!/bin/bash

# Render deployment script
echo "🚀 Streamlit uygulaması başlatılıyor..."

# Gerekli dizinleri oluştur
mkdir -p models
mkdir -p temp

# Python paketlerini güncelle
pip install --upgrade pip

# Requirements'ı yükle
pip install -r requirements.txt

# Streamlit cache dizinini temizle
rm -rf ~/.streamlit

# Streamlit config
mkdir -p ~/.streamlit
cat > ~/.streamlit/config.toml << EOF
[server]
headless = true
port = $PORT
enableCORS = false
enableXsrfProtection = false

[browser]
gatherUsageStats = false
EOF

# Uygulamayı başlat
echo "✅ Kurulum tamamlandı. Uygulama başlatılıyor..."
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
