"""
Aplikasi Penilaian Otomatis + AI + Machine Learning (Tkinter)

Fitur:
- Input data siswa (Nama, Tugas, UTS, UAS)
- Tambah ke tabel
- Simpan rekap ke Excel
- Buat grafik nilai
- Analisis rule-based (remedial, unggulan)
- Train ML (RandomForest) untuk memprediksi Rata-rata dari fitur Tugas & UTS & UAS (atau bisa disesuaikan)
- Predict menggunakan model yang sudah dilatih
- Simpan/load model otomatis (model_rf.joblib)

Simpan file ini sebagai penilaian_gui_ai_ml.py lalu jalankan:
python penilaian_gui_ai_ml.py
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import io

st.set_page_config(page_title="Penilaian Otomatis Guru", page_icon="📊")

st.title("📊 Penilaian Otomatis & AI Prediksi Predikat")

# -------------------------
# 1. Data sementara
# -------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(columns=["Nama", "Tugas", "UTS", "UAS", "Rata-rata", "Predikat"])

# -------------------------
# 2. Form Input Nilai
# -------------------------
with st.form("form_nilai"):
    nama = st.text_input("Nama Siswa")
    tugas = st.number_input("Nilai Tugas", 0, 100, 0)
    uts = st.number_input("Nilai UTS", 0, 100, 0)
    uas = st.number_input("Nilai UAS", 0, 100, 0)
    submit = st.form_submit_button("Tambah Data")
    nama =st.text_input("Nilai Harian")
    harian = st.number_input_("Nilai Harian",0,100,0)

if submit:
    rata = np.mean([tugas, uts, uas,harian])

    # Tentukan predikat manual
    def get_predikat(nilai):
        if nilai >= 85:
            return "A"
        elif nilai >= 75:
            return "B"
        elif nilai >= 65:
            return "C"
        else:
            return "D"

    predikat = get_predikat(rata)

    # Tambahkan ke data
    st.session_state.data = pd.concat([
        st.session_state.data,
        pd.DataFrame([[nama, tugas, uts, uas, rata, predikat]],
                     columns=["Nama", "Tugas", "UTS", "UAS", "Rata-rata", "Predikat"])
    ], ignore_index=True)
    st.success(f"Data untuk {nama} berhasil ditambahkan!")

# -------------------------
# 3. Tampilkan Data Rekap
# -------------------------
st.subheader("📋 Rekap Nilai")

if not st.session_state.data.empty:
    for i, row in st.session_state.data.iterrows():
        col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 1, 1, 1, 1, 1, 1])
        col1.write(row["Nama"])
        col2.write(row["Tugas"])
        col3.write(row["UTS"])
        col4.write(row["UAS"])
        col5.write(row["Rata-rata"])
        col6.write(row["Predikat"])

        # Tombol hapus di kolom terakhir
        hapus = col7.button("🗑 Hapus", key=f"hapus_{i}")
        if hapus:
            st.session_state.data = st.session_state.data.drop(i)
            st.session_state.data.reset_index(drop=True, inplace=True)
            st.success(f"Data '{row['Nama']}' berhasil dihapus!")
            st.experimental_rerun()
else:
    st.warning("Belum ada data.")


# -------------------------
# 4. AI Model (Logistic Regression)
# -------------------------
if len(st.session_state.data) >= 5:
    st.subheader("🤖 AI Prediksi Predikat")
    # Encode predikat ke angka
    label_map = {"A": 3, "B": 2, "C": 1, "D": 0}
    y = st.session_state.data["Predikat"].map(label_map)
    X = st.session_state.data[["Tugas", "UTS", "UAS"]]

    # Latih model
    model = LogisticRegression()
    model.fit(X, y)

    # Input baru untuk prediksi
    st.write("Masukkan nilai untuk memprediksi predikat:")
    tugas_ai = st.number_input("Tugas (AI)", 0, 100, 70, key="tugas_ai")
    uts_ai = st.number_input("UTS (AI)", 0, 100, 70, key="uts_ai")
    uas_ai = st.number_input("UAS (AI)", 0, 100, 70, key="uas_ai")

    if st.button("Prediksi AI"):
        pred_num = model.predict([[tugas_ai, uts_ai, uas_ai]])[0]
        pred_label = {v: k for k, v in label_map.items()}[pred_num]
        st.success(f"Prediksi AI: {pred_label}")

# -------------------------
# 5. Download Rekap
# -------------------------
st.subheader("💾 Download Rekap Nilai")
buffer = io.BytesIO()
with pd.ExcelWriter(buffer, engine="xlsxwriter") as writer:
    st.session_state.data.to_excel(writer, index=False, sheet_name="Rekap Nilai")

st.download_button(
    label="Download Excel",
    data=buffer,
    file_name="rekap_nilai.xlsx",
    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
)

