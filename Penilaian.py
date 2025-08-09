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
    st.session_state.data = pd.DataFrame(columns=["Nama", "Nilai Harian", "Tugas", "UTS", "UAS", "Rata-rata", "Predikat"])

# -------------------------
# 2. Form Input Nilai
# -------------------------
with st.form("form_nilai"):
    nama = st.text_input("Nama Siswa")
    nilai_harian = st.number_input("Nilai Harian", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
    tugas = st.number_input("Nilai Tugas", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
    uts = st.number_input("Nilai UTS", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")
    uas = st.number_input("Nilai UAS", min_value=0.0, max_value=100.0, value=0.0, step=0.1, format="%.1f")

    submit = st.form_submit_button("Tambah Data")

if submit:
    rata = np.mean([nilai_harian, tugas, uts, uas])

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
        pd.DataFrame([[nama, nilai_harian, tugas, uts, uas, rata, predikat]],
                     columns=["Nama", "Nilai Harian", "Tugas", "UTS", "UAS", "Rata-rata", "Predikat"])
    ], ignore_index=True)
    st.success(f"Data untuk {nama} berhasil ditambahkan!")

# -------------------------
# 3. Tampilkan Data Rekap
# -------------------------
st.subheader("📋 Rekap Nilai")

if not st.session_state.data.empty:
    for i, row in st.session_state.data.iterrows():
        col1, col2, col3, col4, col5, col6, col7, col8 = st.columns([2, 1, 1, 1, 1, 1, 1, 1])
        col1.write(row["Nama"])
        col2.write(row["Nilai Harian"])
        col3.write(row["Tugas"])
        col4.write(row["UTS"])
        col5.write(row["UAS"])
        col6.write(row["Rata-rata"])
        col7.write(row["Predikat"])
import io
import datetime

# --- Bagian Download Rekap (paste menggantikan bagian lama) ---
st.subheader("💾 Download Rekap Nilai")

# Salin DataFrame supaya tidak mengubah asli
df_export = st.session_state.data.copy()

if not df_export.empty:
    # (opsional) bulatkan angka ke 1 desimal
    if "Rata-rata" in df_export.columns:
        df_export = df_export.round({col:1 for col in df_export.select_dtypes(include='number').columns})

    # Buat dua kolom untuk tombol CSV dan XLSX
    col_csv, col_xl = st.columns(2)

    # --- CSV ---
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    with col_csv:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"rekap_nilai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )

    # --- XLSX (in-memory) ---
    towrite = io.BytesIO()
    # gunakan engine='xlsxwriter' (pastikan ada di requirements.txt)
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Rekap Nilai")
        # writer.save()  # tidak perlu karena with otomatis menyimpan
    towrite.seek(0)

    with col_xl:
        st.download_button(
            label="⬇️ Download XLSX",
            data=towrite,
            file_name=f"rekap_nilai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
else:
    st.info("Belum ada data yang bisa diunduh.")
    
        # Tombol hapus di kolom terakhir
        hapus = col8.button(
            "🗑 Hapus", 
            key=f"hapus_{i}"
        )
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
    X = st.session_state.data[["Nilai Harian", "Tugas", "UTS", "UAS"]]

    # Latih model
    model = LogisticRegression()
    model.fit(X, y)

    # Input baru untuk prediksi
    st.write("Masukkan nilai untuk memprediksi predikat:")
    nilai_harian_ai = st.number_input("Nilai Harian (AI)", 0, 100, 70, key="nilai_harian_ai")
    tugas_ai = st.number_








