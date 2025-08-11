import streamlit as st
import plotly.express as px
from helper import get_data

df = get_data()

st.title("📊 Şarap Kalitesi Veri Analizi")

# 1. Kalite Dağılımı (Bar Chart)
st.subheader("Şarap Kalite Dağılımı")
quality_counts = df["quality_binary"].value_counts().rename({0: "Düşük Kalite", 1: "Yüksek Kalite"})
fig_quality = px.bar(
    x=quality_counts.index,
    y=quality_counts.values,
    labels={"x": "Kalite Sınıfı", "y": "Şarap Sayısı"},
    text=quality_counts.values,
    title="Şarap Kalite Sınıfına Göre Dağılım"
)
fig_quality.update_traces(textposition="outside")
st.plotly_chart(fig_quality)

st.markdown("---")

# 2. Alkol Seviyesi vs Kalite (Box Plot)
st.subheader("Alkol Miktarının Kaliteye Etkisi")
fig_alcohol_quality = px.box(
    df,
    x="quality_binary",
    y="alcohol",
    labels={"quality_binary": "Kalite (0=Düşük, 1=Yüksek)", "alcohol": "Alkol (%)"},
    title="Alkol Seviyesi ile Şarap Kalitesi Arasındaki Fark"
)
st.plotly_chart(fig_alcohol_quality)

st.markdown("---")

# 3. Korelasyon Isı Haritası
st.subheader("Özellikler Arasındaki Korelasyon")
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
st.plotly_chart(fig_corr)

st.markdown("---")

# 4. pH Kategorileri ve Kalite (Bar Plot)
st.subheader("pH Kategorisi ve Kalite İlişkisi")
if "pH_category" in df.columns:
    ph_quality = df.groupby(["pH_category", "quality_binary"]).size().reset_index(name="count")
    ph_quality["Kalite"] = ph_quality["quality_binary"].map({0: "Düşük", 1: "Yüksek"})

    fig_ph_quality = px.bar(
        ph_quality,
        x="pH_category",
        y="count",
        color="Kalite",
        barmode="group",
        labels={"pH_category": "pH Kategorisi", "count": "Şarap Sayısı"},
        title="pH Kategorilerine Göre Kalite Dağılımı"
    )
    st.plotly_chart(fig_ph_quality)
else:
    st.info("pH kategorileri verisinde eksik.")

st.markdown("---")

# 5. Özellikler Arası Scatterplot (Seçilebilir)
st.subheader("Özellikler Arası İlişki İncelemesi")
cols_for_scatter = numeric_cols.drop("quality_binary") if "quality_binary" in numeric_cols else numeric_cols
x_col = st.selectbox("X Ekseni", options=cols_for_scatter, index=cols_for_scatter.get_loc("alcohol") if "alcohol" in cols_for_scatter else 0)
y_col = st.selectbox("Y Ekseni", options=cols_for_scatter, index=cols_for_scatter.get_loc("fixed acidity") if "fixed acidity" in cols_for_scatter else 1)
color_col = st.selectbox("Renk (Kalite)", options=["None", "quality_binary"], index=1 if "quality_binary" in numeric_cols else 0)

fig_scatter = px.scatter(
    df,
    x=x_col,
    y=y_col,
    color=df[color_col] if color_col != "None" else None,
    labels={x_col: x_col.title(), y_col: y_col.title()},
    title=f"{x_col.title()} vs {y_col.title()}"
)
st.plotly_chart(fig_scatter)
