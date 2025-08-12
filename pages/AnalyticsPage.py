import streamlit as st
import plotly.express as px
import pandas as pd
from helper import get_data

# Sayfa konfigürasyonu
st.set_page_config(
    page_title="Şarap Kalitesi Analizi",
    page_icon="🍷",
    layout="wide"
)

df = get_data()


# Başlık ve kısa açıklama
st.title("🍷 Şarap Kalitesi Veri Analiz Dashboard")
st.markdown("""
Bu sayfada **şarap veri setindeki temel özellikleri** ve **kalite etiketlerini** interaktif grafiklerle inceleyebilirsiniz.
""")

st.subheader("📋 Veri Setinin İlk 5 Satırı")
st.dataframe(df.head())


# -- METRİK KARTLARI --
st.subheader("🔍 Temel İstatistikler")

st.subheader("📌 Temel İstatistikler")

col1, col2, col3, col4 = st.columns(4)
col1.metric("Toplam Kayıt", f"{df.shape[0]}")
col2.metric("Toplam Özellik", f"{df.shape[1]}")
col3.metric("Eksik Değer Sayısı", f"{df.isnull().sum().sum()}")
col4.metric("Kalite Ortalaması", f"{df['quality'].mean():.2f}")

st.markdown("---")

col5, col6, col7, col8 = st.columns(4)
col5.metric("Ortalama Alkol", f"{df['alcohol'].mean():.2f}")
col6.metric("Ortalama pH", f"{df['pH'].mean():.2f}")
col7.metric("Ortalama Fixed Acidity", f"{df['fixed acidity'].mean():.2f}")
col8.metric("Ortalama Residual Sugar", f"{df['residual sugar'].mean():.2f}")

st.markdown("---")

# Binary kalite sınıfı dağılımı
quality_counts = df['quality_binary'].value_counts(normalize=True) * 100
col9, col10 = st.columns(2)
col9.metric("İyi Kalite (%)", f"{quality_counts.get(1, 0):.2f}")
col10.metric("Kötü Kalite (%)", f"{quality_counts.get(0, 0):.2f}")


st.markdown("---")

# -- KALİTE SINIFLARI BAR GRAFİĞİ --
st.subheader("🍇 Şarap Kalite Dağılımı")
quality_counts = df["quality_binary"].value_counts().rename({0: "Düşük Kalite", 1: "Yüksek Kalite"})
fig_quality = px.bar(
    x=quality_counts.index,
    y=quality_counts.values,
    color=quality_counts.index,
    color_discrete_map={"Düşük Kalite": "#d62728", "Yüksek Kalite": "#2ca02c"},
    labels={"x": "Kalite Sınıfı", "y": "Şarap Sayısı"},
    text=quality_counts.values,
    title="Şarap Kalite Sınıfına Göre Dağılım"
)
fig_quality.update_traces(textposition="outside", marker_line_width=0)
fig_quality.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_quality, use_container_width=True)

st.markdown("---")

# -- ALKOL DAĞILIMI BOXPLOT --
st.subheader("🥂 Alkol Seviyesi Dağılımı ve Kalite İlişkisi")
fig_alcohol_quality = px.box(
    df,
    x="quality_binary",
    y="alcohol",
    color="quality_binary",
    color_discrete_map={0: "#d62728", 1: "#2ca02c"},
    labels={"quality_binary": "Kalite (0=Düşük, 1=Yüksek)", "alcohol": "Alkol (%)"},
    title="Alkol Miktarının Kaliteye Etkisi"
)
fig_alcohol_quality.update_layout(showlegend=False, plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_alcohol_quality, use_container_width=True)

st.markdown("---")

# -- KORELASYON ISI HARİTASI --
st.subheader("🔗 Özellikler Arasındaki Korelasyon Matrisi")
numeric_cols = df.select_dtypes(include=["float64", "int64"]).columns
corr = df[numeric_cols].corr()
fig_corr = px.imshow(
    corr,
    labels=dict(x="Özellikler", y="Özellikler", color="Korelasyon"),
    x=corr.columns,
    y=corr.columns,
    color_continuous_scale="RdBu",
    title="Sayısal Özellikler Arası Korelasyon Matrisi"
)
fig_corr.update_layout(plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_corr, use_container_width=True)

st.markdown("---")

# -- INTERAKTİF SCATTERPLOT --
st.subheader("🔎 Özellikler Arası İlişki İncelemesi")
cols_for_scatter = numeric_cols.drop("quality_binary") if "quality_binary" in numeric_cols else numeric_cols
x_col = st.selectbox("X Ekseni", options=cols_for_scatter, index=cols_for_scatter.get_loc("alcohol") if "alcohol" in cols_for_scatter else 0)
y_col = st.selectbox("Y Ekseni", options=cols_for_scatter, index=cols_for_scatter.get_loc("fixed acidity") if "fixed acidity" in cols_for_scatter else 1)
color_col = st.selectbox("Renk (Kalite)", options=["None", "quality_binary"], index=1 if "quality_binary" in numeric_cols else 0)

fig_scatter = px.scatter(
    df,
    x=x_col,
    y=y_col,
    color=df[color_col] if color_col != "None" else None,
    color_continuous_scale=px.colors.sequential.Viridis if color_col == "quality_binary" else None,
    labels={x_col: x_col.title(), y_col: y_col.title()},
    title=f"{x_col.title()} vs {y_col.title()}"
)
fig_scatter.update_layout(plot_bgcolor='rgba(0,0,0,0)')
st.plotly_chart(fig_scatter, use_container_width=True)
