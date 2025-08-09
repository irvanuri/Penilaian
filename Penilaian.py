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

import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
import pandas as pd
import matplotlib.pyplot as plt
import statistics
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
import joblib

# ------------------------------
# Data sementara (in-memory)
# ------------------------------
data_siswa = []  # tiap elemen: [Nama, Tugas, UTS, UAS, Rata-rata, Predikat]

MODEL_PATH = "model_rf.joblib"

# ------------------------------
# Fungsi util
# ------------------------------
def predikat(nilai):
    if nilai >= 85:
        return "A"
    elif nilai >= 75:
        return "B"
    elif nilai >= 65:
        return "C"
    else:
        return "D"

def update_table():
    # Clear tabel lalu isi ulang dari data_siswa
    for row in tabel.get_children():
        tabel.delete(row)
    for row in data_siswa:
        tabel.insert("", tk.END, values=(row[0], row[1], row[2], row[3], round(row[4],2), row[5]))

# ------------------------------
# Fungsi Tambah Data
# ------------------------------
def tambah_data():
    try:
        nama = entry_nama.get().strip()
        if not nama:
            messagebox.showerror("Error", "Nama harus diisi.")
            return
        tugas = float(entry_tugas.get())
        uts = float(entry_uts.get())
        uas = float(entry_uas.get())

        rata2 = (tugas + uts + uas) / 3
        grade = predikat(rata2)

        data_siswa.append([nama, tugas, uts, uas, rata2, grade])
        update_table()

        entry_nama.delete(0, tk.END)
        entry_tugas.delete(0, tk.END)
        entry_uts.delete(0, tk.END)
        entry_uas.delete(0, tk.END)
    except ValueError:
        messagebox.showerror("Error", "Masukkan nilai numerik untuk Tugas/UTS/UAS.")

# ------------------------------
# Fungsi Simpan ke Excel
# ------------------------------
def simpan_excel():
    if not data_siswa:
        messagebox.showerror("Error", "Tidak ada data untuk disimpan!")
        return
    df = pd.DataFrame(data_siswa, columns=["Nama","Tugas","UTS","UAS","Rata-rata","Predikat"])
    filename = "rekap_nilai_gui_ai_ml.xlsx"
    df.to_excel(filename, index=False)
    messagebox.showinfo("Berhasil", f"Data berhasil disimpan ke {filename}")

# ------------------------------
# Fungsi Buat Grafik
# ------------------------------
def buat_grafik():
    if not data_siswa:
        messagebox.showerror("Error", "Tidak ada data untuk digrafikkan!")
        return
    df = pd.DataFrame(data_siswa, columns=["Nama","Tugas","UTS","UAS","Rata-rata","Predikat"])
    plt.figure(figsize=(9,6))
    for i in range(len(df)):
        plt.plot(["Tugas","UTS","UAS"], df.iloc[i,1:4], marker='o', label=df.iloc[i,0])
    plt.title("Grafik Nilai Siswa")
    plt.xlabel("Jenis Penilaian")
    plt.ylabel("Nilai")
    plt.legend(bbox_to_anchor=(1.05,1), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("grafik_nilai_gui_ai_ml.png")
    plt.show()
    messagebox.showinfo("Selesai", "Grafik tersimpan sebagai grafik_nilai_gui_ai_ml.png")

# ------------------------------
# Fungsi Analisis AI (rule-based)
# ------------------------------
def analisis_ai():
    if not data_siswa:
        messagebox.showerror("Error", "Tidak ada data untuk dianalisis!")
        return
    df = pd.DataFrame(data_siswa, columns=["Nama","Tugas","UTS","UAS","Rata-rata","Predikat"])
    rata_kelas = statistics.mean(df["Rata-rata"])
    siswa_rendah = df[df["Rata-rata"] < 65]["Nama"].tolist()
    siswa_tinggi = df[df["Rata-rata"] >= 85]["Nama"].tolist()

    rekomendasi = f"📊 Analisis AI Nilai Siswa\n"
    rekomendasi += f"- Rata-rata kelas: {rata_kelas:.2f}\n"
    rekomendasi += f"- Siswa berpotensi remedial: {', '.join(siswa_rendah) if siswa_rendah else 'Tidak ada'}\n"
    rekomendasi += f"- Siswa unggulan: {', '.join(siswa_tinggi) if siswa_tinggi else 'Tidak ada'}\n\n"

    if siswa_rendah:
        rekomendasi += "💡 Saran untuk remedial:\n- Bimbingan tambahan one-on-one.\n- Latihan soal bertahap (mulai dari konsep dasar).\n- Gunakan media interaktif (video, kuis).\n\n"
    if siswa_tinggi:
        rekomendasi += "💡 Saran untuk siswa unggulan:\n- Berikan tugas proyek/ekstra.\n- Ajak ikut kompetisi atau kelompok studi.\n\n"
    rekomendasi += "ℹ️ Catatan: Analisis rule-based berguna cepat; gunakan hasil ML untuk prediksi lebih akurat jika tersedia model."
    messagebox.showinfo("Analisis AI", rekomendasi)

# ------------------------------
# Fungsi Machine Learning: Train
# ------------------------------
def train_model():
    if len(data_siswa) < 10:
        # Jumlah minimal disarankan 10, tapi bisa diubah
        ok = messagebox.askyesno("Peringatan", "Disarankan minimal 10 data untuk melatih model. Tetap lanjutkan?")
        if not ok:
            return
    df = pd.DataFrame(data_siswa, columns=["Nama","Tugas","UTS","UAS","Rata-rata","Predikat"])
    # Fitur dan target: gunakan Tugas, UTS, UAS untuk memprediksi Rata-rata
    X = df[["Tugas","UTS","UAS"]].values
    y = df["Rata-rata"].values

    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Pipeline scaler + model
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(n_estimators=100, random_state=42))
    ])

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    # Simpan model
    joblib.dump(pipeline, MODEL_PATH)

    messagebox.showinfo("Training Selesai", f"Model telah dilatih dan disimpan ke {MODEL_PATH}\nMAE: {mae:.2f}\nR²: {r2:.2f}")

# ------------------------------
# Fungsi Predict (menggunakan model jika ada)
# ------------------------------
def predict_for_selected_or_input():
    # Ambil model
    if not os.path.exists(MODEL_PATH):
        messagebox.showerror("Error", f"Tidak ditemukan model terlatih ({MODEL_PATH}). Silakan klik 'Train Model' dahulu.")
        return
    model = joblib.load(MODEL_PATH)

    # Cek apakah ada baris terpilih di tabel
    sel = tabel.selection()
    if sel:
        vals = tabel.item(sel[0])["values"]
        # vals = (Nama, Tugas, UTS, UAS, Rata-rata, Predikat)
        tugas = float(vals[1])
        uts = float(vals[2])
        uas = float(vals[3])
        siswa_nama = vals[0]
    else:
        # Minta input sederhana dari user
        siswa_nama = simpledialog.askstring("Input", "Nama siswa untuk diprediksi:")
        if not siswa_nama:
            return
        try:
            tugas = float(simpledialog.askstring("Input", "Nilai Tugas:"))
            uts = float(simpledialog.askstring("Input", "Nilai UTS:"))
            uas = float(simpledialog.askstring("Input", "Nilai UAS:"))
        except (TypeError, ValueError):
            messagebox.showerror("Error", "Input nilai tidak valid.")
            return

    X_new = [[tugas, uts, uas]]
    pred_rata = model.predict(X_new)[0]
    pred_grade = predikat(pred_rata)

    rekom = f"Hasil prediksi untuk {siswa_nama}:\n- Rata-rata terprediksi: {pred_rata:.2f}\n- Predikat terprediksi: {pred_grade}\n"
    # Rekomendasi berdasarkan prediksi
    if pred_rata < 65:
        rekom += "\nSaran: Pertimbangkan bimbingan remedial, latihan soal dasar, dan monitoring berkala."
    elif pred_rata < 75:
        rekom += "\nSaran: Latihan terfokus pada topik lemah, kuis mingguan."
    elif pred_rata >= 85:
        rekom += "\nSaran: Berikan tantangan tambah, materi enrichment."
    else:
        rekom += "\nSaran: Pertahankan metode yang berjalan, tambah latihan soal aplikasi."

    messagebox.showinfo("Prediksi ML", rekom)

# ------------------------------
# Fungsi Load contoh data (opsional) - membantu demo
# ------------------------------
def load_sample_data():
    # Hanya jika tabel kosong, untuk demo cepat
    if data_siswa:
        ok = messagebox.askyesno("Konfirmasi", "Data sudah ada. Ingin menambahkan sample demo? (akan menambah baris baru)")
        if not ok:
            return
    sample = [
        ["Andi", 85, 80, 88],
        ["Budi", 78, 75, 82],
        ["Citra", 90, 85, 87],
        ["Dewi", 88, 90, 92],
        ["Eko", 70, 65, 68],
        ["Fani", 60, 55, 62],
        ["Gilang", 72, 70, 75],
        ["Hesti", 95, 92, 94],
        ["Irwan", 67, 68, 66],
        ["Joko", 80, 78, 79],
        ["Kiki", 83, 81, 86],
        ["Lia", 74, 70, 73],
    ]
    for nama, t, u, ua in sample:
        rata = (t+u+ua)/3
        data_siswa.append([nama, t, u, ua, rata, predikat(rata)])
    update_table()
    messagebox.showinfo("Sample", "Sample data ditambahkan (12 baris). Anda bisa langsung klik Train Model.")

# ------------------------------
# Tampilan Tkinter
# ------------------------------
root = tk.Tk()
root.title("Aplikasi Penilaian Otomatis + AI + ML")
root.geometry("900x600")

# Input Form
frame_input = tk.Frame(root)
frame_input.pack(pady=8, anchor="w")

tk.Label(frame_input, text="Nama Siswa").grid(row=0, column=0, padx=4)
tk.Label(frame_input, text="Tugas").grid(row=0, column=1, padx=4)
tk.Label(frame_input, text="UTS").grid(row=0, column=2, padx=4)
tk.Label(frame_input, text="UAS").grid(row=0, column=3, padx=4)

entry_nama = tk.Entry(frame_input, width=25)
entry_tugas = tk.Entry(frame_input, width=8)
entry_uts = tk.Entry(frame_input, width=8)
entry_uas = tk.Entry(frame_input, width=8)

entry_nama.grid(row=1, column=0, padx=4)
entry_tugas.grid(row=1, column=1, padx=4)
entry_uts.grid(row=1, column=2, padx=4)
entry_uas.grid(row=1, column=3, padx=4)

btn_tambah = tk.Button(frame_input, text="Tambah Data", command=tambah_data)
btn_tambah.grid(row=1, column=4, padx=8)

btn_sample = tk.Button(frame_input, text="Load Sample Data", command=load_sample_data)
btn_sample.grid(row=1, column=5, padx=8)

# Tabel
kolom = ("Nama","Tugas","UTS","UAS","Rata-rata","Predikat")
tabel = ttk.Treeview(root, columns=kolom, show="headings", selectmode="browse", height=15)
for col in kolom:
    tabel.heading(col, text=col)
    # Atur lebar kolom
    if col == "Nama":
        tabel.column(col, width=180, anchor="w")
    else:
        tabel.column(col, width=90, anchor="center")
tabel.pack(pady=8, fill="x")

# Tombol aksi
frame_btn = tk.Frame(root)
frame_btn.pack(pady=6)

btn_simpan = tk.Button(frame_btn, text="Simpan ke Excel", command=simpan_excel, width=15)
btn_simpan.grid(row=0, column=0, padx=6)

btn_grafik = tk.Button(frame_btn, text="Buat Grafik", command=buat_grafik, width=15)
btn_grafik.grid(row=0, column=1, padx=6)

btn_ai = tk.Button(frame_btn, text="Analisis AI (Rule-Based)", command=analisis_ai, width=18, bg="lightblue")
btn_ai.grid(row=0, column=2, padx=6)

btn_train = tk.Button(frame_btn, text="Train Model (ML)", command=train_model, width=15, bg="lightgreen")
btn_train.grid(row=0, column=3, padx=6)

btn_predict = tk.Button(frame_btn, text="Predict (ML)", command=predict_for_selected_or_input, width=15, bg="lightyellow")
btn_predict.grid(row=0, column=4, padx=6)

# Status / info
status_var = tk.StringVar()
status_var.set("Status: Ready")
label_status = tk.Label(root, textvariable=status_var, anchor="w")
label_status.pack(fill="x", padx=8, pady=(0,8))

# Saat aplikasi mulai, jika model ada, beri tahu
if os.path.exists(MODEL_PATH):
    status_var.set(f"Status: Model ditemukan di {MODEL_PATH} (siap dipakai).")
else:
    status_var.set("Status: Model belum dilatih. Klik 'Load Sample Data' lalu 'Train Model' untuk demo cepat.")

root.mainloop()
