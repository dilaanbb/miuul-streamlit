import streamlit as st
import os
import numpy as np
import joblib
from helper import feature_names

# Dosya yollarını ayarla
current_dir = os.path.dirname(os.path.abspath(__file__))
base_dir = os.path.dirname(current_dir)
models_dir = os.path.join(base_dir, "models")

model_path = os.path.join(models_dir, "best_wine_quality_model.pkl")
scaler_path = os.path.join(models_dir, "scaler.pkl")
le_path = os.path.join(models_dir, "label_encoder.pkl")

# Model, scaler ve label encoder dosyalarını yükle
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
le = joblib.load(le_path)

st.title("🍷 Şarap Kalitesi Tahmin")

# Kullanıcıdan girdileri al
inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature.title()}", value=0.0)
    inputs.append(val)

if st.button("Tahmin Et"):
    st.toast('Tahmin Başarıyla Gerçekleşti!', icon='🎉')
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    pred_encoded = model.predict(input_scaled)[0]
    pred_label = le.inverse_transform([pred_encoded])[0]
    st.success(f"Şarap kalitesi tahmini: {pred_label.replace('_', ' ').title()}")




