import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# Yalnızca ilk seferlik veriler yükleniyor
patients = pd.read_csv("patients.csv")
doctors = pd.read_csv("doctors.csv")
appointments = pd.read_csv("appointments.csv")

# Sadece Completed (tamamlanmış) randevular
completed_appointments = appointments[appointments["Status"] == "Completed"]

# Hasta-Doktor matrisi
user_doctor_matrix = completed_appointments.pivot_table(
    index='Patient_ID',
    columns='Doctor_ID',
    values='Doctor_Choice_Num',
    aggfunc='mean'
).fillna(0)

# Benzerlik matrisi
user_similarity = cosine_similarity(user_doctor_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_doctor_matrix.index, columns=user_doctor_matrix.index)

# Doktor önerme fonksiyonu
def recommend_doctor(patient_id, specialty_filter=None, top_n=3):
    if patient_id not in user_doctor_matrix.index:
        return "Hasta bulunamadı."

    similar_patients = user_similarity_df[patient_id].sort_values(ascending=False)[1:]

    recommended_doctors = pd.Series(dtype=float)

    for other_patient, similarity in similar_patients.items():
        patient_ratings = user_doctor_matrix.loc[other_patient]
        patient_ratings = patient_ratings[patient_ratings == 1]
        recommended_doctors = pd.concat([recommended_doctors, patient_ratings * similarity])

    existing_doctors = user_doctor_matrix.loc[patient_id][user_doctor_matrix.loc[patient_id] == 1].index.tolist()
    recommended_doctors = recommended_doctors.drop(labels=existing_doctors, errors='ignore')

    recommended_doctors = recommended_doctors.groupby(recommended_doctors.index).sum().sort_values(ascending=False)

    recommended_doctors_df = pd.DataFrame({
        'Doctor_ID': recommended_doctors.index,
        'Score': recommended_doctors.values
    }).merge(doctors, on='Doctor_ID')

    if specialty_filter:
        recommended_doctors_df = recommended_doctors_df[recommended_doctors_df['Specialty'] == specialty_filter]

    return recommended_doctors_df[['Doctor_ID', 'Doctor_Name', 'Specialty', 'Hospital', 'Score']].head(top_n)
