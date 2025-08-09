import streamlit as st


home_page = st.Page(page = "pages/HomePage.py", title="QualiWine",icon = "🏠")

analytics_page = st.Page(page = "pages/AnalyticsPage.py",title = "Dataset",icon = "📊")

predict_page = st.Page(page = "pages/PredictPage.py",title = "Prediction",icon = "️⁉️")


pg = st.navigation([home_page,analytics_page,predict_page])
st.set_page_config(layout = "wide",page_title="QualiWine",page_icon="🍷")

pg.run()