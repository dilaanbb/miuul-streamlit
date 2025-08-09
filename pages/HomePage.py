import streamlit as st

st.set_page_config(page_title = "QualiWine",page_icon="🍷",layout = "wide")

st.title(":red[Quali]:green[Wine]")
st.subheader("Şarap özelliklerini girin, kalitesini öğrenin.")


left_col,right_col = st.columns(2)

left_col.markdown("""
🎯 **Bu uygulama tam sana göre!**

Aşağıdaki kutucuklara şaraba ait bazı temel bilgileri (örneğin asit oranı, alkol seviyesi gibi) giriyorsun, biz de sana bu şarabın kalitesinin **düşük mü, orta mı yoksa yüksek mi** olduğunu tahmin ediyoruz.  

Yani bir nevi _küçük bir tadım uzmanı_ gibi düşün! 😊  
Hem hızlı, hem eğlenceli, hem de oldukça basit.

Kendi verilerinle denemeler yapabilir, farklı değerlerin kaliteyi nasıl etkilediğini görebilirsin.  
**Hazırsan başlayalım!** 🍷✨
""")

right_col.image("vinho_verde.jpg")
