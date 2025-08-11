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

# Model ve scaler dosyalarını yükle
model = joblib.load(model_path)
scaler = joblib.load(scaler_path)

st.title("🍷 Şarap Kalitesi Tahmin")

# Tahmin sonucunu etiketlere çeviren fonksiyon
def map_prediction_to_label(pred_int):
    label_map = {0: "Low Quality", 1: "High Quality"}
    return label_map.get(pred_int, "Unknown")

inputs = []
for feature in feature_names:
    val = st.number_input(f"{feature.title()}", value=0.0, format="%.3f")
    inputs.append(val)

if st.button("Tahmin Et"):
    input_array = np.array([inputs])
    input_scaled = scaler.transform(input_array)
    pred_encoded = model.predict(input_scaled)[0]
    pred_label = map_prediction_to_label(pred_encoded)
    st.success(f"Şarap kalitesi tahmini: {pred_label}")
