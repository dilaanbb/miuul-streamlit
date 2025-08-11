import pandas as pd
import streamlit as st

# Özellik adları
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Veri setini yükleyip işleyen yardımcı fonksiyon
@st.cache_data
def get_data():
    df = pd.read_csv("winequality-red-processed.csv")
    df.dropna(inplace=True)  # Eksik verileri kaldır
    return df

# Tahmin değerini etikete dönüştüren yardımcı fonksiyon (binary versiyon)
def map_prediction_to_label(pred_int):
    label_map = {0: "Low Quality", 1: "High Quality"}
    return label_map.get(pred_int, "Unknown")
