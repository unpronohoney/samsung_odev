# 📌 1. Gerekli kütüphaneler
import importlib
import pandas as pd
import streamlit as st
import os

# 📌 2. .py dosyaları varsa bir kez çalıştır / içe aktar
if os.path.exists("recommendation_engine.py"):
    recommendation_engine = importlib.import_module("recommendation_engine")
else:
    raise FileNotFoundError("recommendation_engine.py dosyası eksik!")

if os.path.exists("model_utils.py"):
    speciality_recommend = importlib.import_module("model_utils")
else:
    raise FileNotFoundError("model_utils.py dosyası eksik!")

# 📌 3. Streamlit Arayüzü
st.set_page_config(page_title="Akıllı Doktor Öneri Sistemi", layout="centered")

st.title("🏥 Akıllı Doktor Öneri ve Branş Tahmin Sistemi")

# 🟦 4. Şikayet Alanı (BERT modeli çalıştırmak için)
st.subheader("🔎 Şikayetinizi Yazın")
complaint = st.text_area("Lütfen sağlık şikayetinizi detaylı şekilde yazın...")

if st.button("Branşı Tahmin Et (Yapay Zeka)"):
    predicted_specialty = speciality_recommend.predict_specialty(complaint)
    st.success(f"🔍 Tahmin Edilen Branş: **{predicted_specialty}**")

# 🟨 5. Doktor Önerisi Alanı
st.subheader("🩺 Doktor Önerisi Al")

patient_id = st.number_input("Hasta ID'nizi girin", min_value=0, step=1)

# Uzmanlık listesi sabit veya dinamik olabilir
specialty_options = [
    "Dahiliye", "Jinekoloji", "Pediatri", "Göz Hastalıkları", "Kulak Burun Boğaz",
    "Cildiye", "Kardiyoloji", "Ortopodi", "Nöroloji", "Psikiyatri", "Üroloji", "Genel Cerrahi"
]

specialty_choice = st.selectbox("Tercih ettiğiniz branş (isteğe bağlı)", [""] + specialty_options)

if st.button("👨‍⚕️ Doktor Öner"):
    try:
        filtered_specialty = specialty_choice if specialty_choice else None
        results = recommendation_engine.recommend_doctor(patient_id, specialty_filter=filtered_specialty, top_n=5)

        if isinstance(results, str):
            st.error(results)
        else:
            st.subheader("✅ Önerilen Doktorlar")
            st.dataframe(results)

    except Exception as e:
        st.error(f"❌ Hata oluştu: {e}")
