import pandas as pd
import streamlit as st

# Kaliteyi sınıflara ayıran harita
mapping = {
    3: "low_quality",
    4: "low_quality",
    5: "medium_quality",
    6: "medium_quality",
    7: "high_quality",
    8: "high_quality"
}

# Özellik adları
feature_names = [
    'fixed acidity', 'volatile acidity', 'citric acid', 'residual sugar',
    'chlorides', 'free sulfur dioxide', 'total sulfur dioxide', 'density',
    'pH', 'sulphates', 'alcohol'
]

# Veri setini yükleyip işleyen yardımcı fonksiyon
@st.cache_data
def get_data():
    df = pd.read_csv("winequality-red.csv")
    df.dropna(inplace=True)  # Eksik verileri at
    df["quality_cat"] = df["quality"].map(mapping)  # Kalite kategorisi ekle
    return df

# Kalite değerini string etikete dönüştüren yardımcı fonksiyon (opsiyonel)
def map_prediction_to_label(pred_int):
    label_map = ["low_quality", "medium_quality", "high_quality"]
    if 0 <= pred_int < len(label_map):
        return label_map[pred_int]
    return "unknown"



