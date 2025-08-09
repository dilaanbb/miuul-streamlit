import os
import sys
import plotly.express as px
from helper import get_data
import streamlit as st

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


df = get_data()

st.title("📊 Analiz Sayfası")
st.write("Veri seti aşağıda görüntülenmektedir:")
st.dataframe(df)

avg_fixed_acidity = df["fixed acidity"].mean()
avg_volatile_acidity = df["volatile acidity"].mean()
avg_citric_acid = df["citric acid"].mean()
avg_residual_sugar = df["residual sugar"].mean()
avg_chlorides = df["chlorides"].mean()
avg_free_sulfur_dioxide = df["free sulfur dioxide"].mean()
avg_total_sulfur_dioxide = df["total sulfur dioxide"].mean()
avg_density = df["density"].mean()
avg_pH = df["pH"].mean()
avg_sulphates = df["sulphates"].mean()
avg_alcohol = df["alcohol"].mean()

st.subheader("📌Temel Öznitelikler")

col1,col2,col3,col4,\
    col5,col6,col7,col8,col9,col10,col11= st.columns(11)
col1.metric("Average Fixed Acidity",f"{avg_fixed_acidity:.2f}")
col2.metric("Average Volatile Acidity",f"{avg_volatile_acidity:.2f}")
col3.metric("Average Citric Acid",f"{avg_citric_acid:.2f}")
col4.metric("Average Residual Sugar",f"{avg_residual_sugar:.2f}")
col5.metric("Average Chlorides",f"{avg_chlorides:.2f}")
col6.metric("Average Free Sulfur Dioxide",f"{avg_free_sulfur_dioxide:.2f}")
col7.metric("Average Total Sulfur Dioxide",f"{avg_total_sulfur_dioxide:.2f}")
col8.metric("Average Density",f"{avg_density:.2f}")
col9.metric("Average pH",f"{avg_pH:.2f}")
col10.metric("Average Sulphates",f"{avg_sulphates:.2f}")
col11.metric("Average Alcohol",f"{avg_alcohol:.2f}")

st.markdown("---")

# Scatterplot
# Scatterplot
st.subheader("Scatterplot: Değişkenler Arası İlişki")
x_var = st.selectbox(
    "X ekseni değişkeni",
    df.columns,
    index=df.columns.get_loc("alcohol"),
    key="scatter_x"
)
y_var = st.selectbox(
    "Y ekseni değişkeni",
    df.columns,
    index=df.columns.get_loc("fixed acidity"),
    key="scatter_y"
)
color_var = st.selectbox(
    "Renk (opsiyonel)",
    [None] + list(df.columns),
    index=1,
    key="scatter_color"
)

fig_scatter = px.scatter(
    df,
    x=x_var,
    y=y_var,
    color=color_var if color_var else None,
    hover_data=[x_var, y_var],
    title=f"{x_var} vs {y_var}"
)
st.plotly_chart(fig_scatter)


st.markdown("---")

# Histogram bölümü
st.subheader("Histogram: Dağılım Analizi")
hist_var = st.selectbox(
    "Histogram değişkeni",
    df.columns,
    index=df.columns.get_loc("alcohol"),
    key="hist_var"
)
color_var_hist = st.selectbox(
    "Renk (opsiyonel)",
    [None] + list(df.columns),
    index=1,
    key="hist_color"
)
bins = st.slider(
    "Bin sayısı",
    5,
    50,
    20,
    key="hist_bins"
)

fig_hist = px.histogram(
    df,
    x=hist_var,
    color=color_var_hist if color_var_hist else None,
    nbins=bins,
    marginal="box",
    title=f"{hist_var} Dağılımı"
)
st.plotly_chart(fig_hist)
