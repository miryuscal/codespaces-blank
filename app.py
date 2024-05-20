import streamlit as st
import os
import numpy as np
import pandas as pd
import cv2
from tensorflow.keras.models import load_model

# Sınıf eşlemesini ve diğer gerekli fonksiyonları burada tanımlayın

# Modelinizi yükleyin
model = load_model('C:/Users/Yusuf/Desktop/AracTanimaSistemi/model.h5')

# Streamlit uygulamasını oluşturun
st.title('Car Brand Classification')

# Görüntü yükleme bölümü
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Yüklenen görüntüyü işleyin
    image = cv2.imdecode(np.fromstring(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Görüntüyü modelinizin girdi boyutuna yeniden boyutlandırın
    image = cv2.resize(image, (224, 224))
    image = np.array(image) / 255.0  # Normalizasyon
    
    # Sınıflandırma yapın
    result = model.predict(np.expand_dims(image, axis=0))
    
    # Sonuçları gösterin
    st.write("Prediction Probabilities:")
    st.write({
        "Volkswagen": result[0][0],
        "Toyota": result[0][1],
        "BMW": result[0][2],
        "Hyundai": result[0][3],
        "MercedesBenz": result[0][4]
    })
