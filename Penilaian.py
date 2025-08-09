# Penilaian.py
import streamlit as st
import pandas as pd
import numpy as np
import io
import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Penilaian Otomatis Guru", page_icon="📊", layout="wide")
st.title("📊 Aplikasi Penilaian Otomatis (Streamlit)")

# -------------------------
# Helper
# -------------------------
def get_predikat(nilai):
    if nilai >= 85:
        return "A"
    elif nilai >= 75:
        return "B"
    elif nilai >= 65:
        return "C"
    else:
        return "D"

# -------------------------
# Inisialisasi session_state
# -------------------------
if "data" not in st.session_state:
    st.session_state.data = pd.DataFrame(
        columns=["Nama", "Nilai Harian", "Tugas", "UTS", "UAS", "Rata-rata", "Predikat"]
    )

# -------------------------
# Form input (nilai bisa desimal 1 tempat)
# -------------------------
with st.form("form_nilai"):
    col_a, col_b = st.columns([3, 5])
    with col_a:
        nama = st.text_input("Nama Siswa", key="input_nama")
    with col_b:
        c1, c2, c3, c4 = st.columns(4)
        nilai_harian = c1.number_input(
            "Nilai Harian", min_value=0.0, max_value=100.0, value=0.0,
            step=0.1, format="%.1f", key="input_nh"
        )
        tugas = c2.number_input(
            "Nilai Tugas", min_value=0.0, max_value=100.0, value=0.0,
            step=0.1, format="%.1f", key="input_tugas"
        )
        uts = c3.number_input(
            "Nilai UTS", min_value=0.0, max_value=100.0, value=0.0,
            step=0.1, format="%.1f", key="input_uts"
        )
        uas = c4.number_input(
            "Nilai UAS", min_value=0.0, max_value=100.0, value=0.0,
            step=0.1, format="%.1f", key="input_uas"
        )

    submit = st.form_submit_button("Tambah Data")

if submit:
    if not nama or not nama.strip():
        st.error("Nama siswa harus diisi.")
    else:
        # pastikan satu angka di belakang koma
        nh = round(float(nilai_harian), 1)
        tgs = round(float(tugas), 1)
        uts_v = round(float(uts), 1)
        uas_v = round(float(uas), 1)

        rata = round(np.mean([nh, tgs, uts_v, uas_v]), 1)
        pred = get_predikat(rata)

        row = pd.DataFrame([[nama.strip(), nh, tgs, uts_v, uas_v, rata, pred]],
                           columns=st.session_state.data.columns)
        st.session_state.data = pd.concat([st.session_state.data, row], ignore_index=True)
        st.success(f"Data untuk '{nama.strip()}' berhasil ditambahkan!")

# -------------------------
# Tampilkan Rekap dengan tombol Hapus di sebelah kanan
# -------------------------
st.subheader("📋 Rekap Nilai")

df = st.session_state.data

if not df.empty:
    # Header
    hcols = st.columns([2,1,1,1,1,1,1,1])
    hcols[0].markdown("**Nama**")
    hcols[1].markdown("**Nilai Harian**")
    hcols[2].markdown("**Tugas**")
    hcols[3].markdown("**UTS**")
    hcols[4].markdown("**UAS**")
    hcols[5].markdown("**Rata-rata**")
    hcols[6].markdown("**Predikat**")
    hcols[7].markdown("**Aksi**")

    # Baris data
    for i, row in df.iterrows():
        c0, c1, c2, c3, c4, c5, c6, c7 = st.columns([2,1,1,1,1,1,1,1])
        c0.write(row["Nama"])
        # tampilkan dengan 1 desimal
        c1.write(f"{row['Nilai Harian']:.1f}")
        c2.write(f"{row['Tugas']:.1f}")
        c3.write(f"{row['UTS']:.1f}")
        c4.write(f"{row['UAS']:.1f}")
        c5.write(f"{row['Rata-rata']:.1f}")
        c6.write(row["Predikat"])

        # tombol hapus unik per baris
        if c7.button("🗑 Hapus", key=f"hapus_{i}"):
            st.session_state.data = st.session_state.data.drop(i).reset_index(drop=True)
            st.success(f"Data '{row['Nama']}' berhasil dihapus.")
            st.experimental_rerun()
else:
    st.info("Belum ada data.")

# -------------------------
# AI: train & predict (classify predikat) — otomatis jika data >= 5
# -------------------------
if len(st.session_state.data) >= 5:
    st.subheader("🤖 AI Prediksi Predikat (Logistic Regression)")
    features = ["Nilai Harian", "Tugas", "UTS", "UAS"]
    label_map = {"A": 3, "B": 2, "C": 1, "D": 0}

    df_ml = st.session_state.data.copy()
    X = df_ml[features].values
    y = df_ml["Predikat"].map(label_map).values

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    st.write(f"Model akurasi (test set): **{acc:.2f}** ({len(df_ml)} baris data)")

    # Input manual untuk prediksi
    st.markdown("**Coba Prediksi untuk Nilai Baru**")
    p1, p2, p3, p4 = st.columns(4)
    ph = p1.number_input("Nilai Harian (AI)", min_value=0.0, max_value=100.0, value=70.0,
                        step=0.1, format="%.1f", key="ai_nh")
    pt = p2.number_input("Tugas (AI)", min_value=0.0, max_value=100.0, value=70.0,
                         step=0.1, format="%.1f", key="ai_t")
    puts = p3.number_input("UTS (AI)", min_value=0.0, max_value=100.0, value=70.0,
                           step=0.1, format="%.1f", key="ai_uts")
    puas = p4.number_input("UAS (AI)", min_value=0.0, max_value=100.0, value=70.0,
                           step=0.1, format="%.1f", key="ai_uas")

    if st.button("Prediksi AI"):
        X_new = np.array([[round(ph,1), round(pt,1), round(puts,1), round(puas,1)]])
        pred_num = model.predict(X_new)[0]
        inv_map = {v:k for k,v in label_map.items()}
        pred_label = inv_map.get(pred_num, "?")
        st.success(f"Prediksi AI → Predikat: **{pred_label}**")

# -------------------------
# Download CSV & XLSX
# -------------------------
st.subheader("💾 Download Rekap Nilai")
df_export = st.session_state.data.copy()
if not df_export.empty:
    # bulatkan angka numerik ke 1 desimal
    num_cols = df_export.select_dtypes(include="number").columns
    df_export[num_cols] = df_export[num_cols].round(1)

    col_csv, col_xl = st.columns(2)

    # CSV
    csv_bytes = df_export.to_csv(index=False).encode("utf-8")
    with col_csv:
        st.download_button(
            label="⬇️ Download CSV",
            data=csv_bytes,
            file_name=f"rekap_nilai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    # XLSX (in-memory)
    towrite = io.BytesIO()
    with pd.ExcelWriter(towrite, engine="xlsxwriter") as writer:
        df_export.to_excel(writer, index=False, sheet_name="Rekap Nilai")
    towrite.seek(0)
    with col_xl:
        st.download_button(
            label="⬇️ Download XLSX",
            data=towrite,
            file_name=f"rekap_nilai_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
else:
    st.info("Belum ada data yang bisa diunduh.")
