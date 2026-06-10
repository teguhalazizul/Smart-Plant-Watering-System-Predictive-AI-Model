# 🌱 Smart Plant Watering System — Model Prediksi AI

> Proyek machine learning untuk memprediksi apakah tanaman perlu disiram atau tidak berdasarkan data sensor lingkungan, menggunakan **Random Forest Classifier**.

---

## 📌 Deskripsi Proyek

Proyek ini membangun model klasifikasi untuk menentukan status penyiraman tanaman (`ON` / `OFF`) secara otomatis berdasarkan kondisi tanah dan lingkungan sekitar. Model dilatih menggunakan data sensor nyata dan dievaluasi dengan berbagai metrik untuk memastikan keandalannya.

**Tujuan:** Sistem irigasi cerdas — mengotomatiskan keputusan penyiraman tanaman menggunakan AI, tanpa perlu pengecekan manual.

---

## 📂 Dataset

- **Sumber:** [Kaggle — Dataset for Predicting Watering the Plants](https://www.kaggle.com/datasets/nelakurthisudheer/dataset-for-predicting-watering-the-plants)
- **File:** `TARP.csv`
- **Kolom target:** `Status` (ON / OFF)

### Fitur yang Digunakan

| Fitur | Keterangan |
|-------|------------|
| `Soil Moisture` | Tingkat kelembapan tanah |
| `Temperature` | Suhu lingkungan |
| `Soil Humidity` | Kelembapan tanah |
| `Air temperature (C)` | Suhu udara dalam Celsius |
| `Wind speed (Km/h)` | Kecepatan angin |
| `Air humidity (%)` | Persentase kelembapan udara |
| `Wind gust (Km/h)` | Kecepatan hembusan angin |
| `Pressure (KPa)` | Tekanan atmosfer |
| `ph` | Tingkat pH tanah |
| `rainfall` | Jumlah curah hujan |
| `N` | Kandungan Nitrogen dalam tanah |
| `P` | Kandungan Fosfor dalam tanah |
| `K` | Kandungan Kalium dalam tanah |

---

## 🔄 Alur Kerja

```
Dataset (TARP.csv)
      │
      ▼
Pra-pemrosesan Data
  ├── Menghapus baris yang bernilai null
  ├── Label Encoding (Status: ON → 1, OFF → 0)
  └── Pemisahan fitur dan target
      │
      ▼
Analisis Data Eksploratif (EDA)
  ├── Histogram distribusi fitur
  ├── Boxplot (deteksi outlier)
  └── Heatmap korelasi antar fitur
      │
      ▼
Pelatihan Model
  └── Random Forest Classifier
        ├── n_estimators = 100
        ├── max_features = 'sqrt'
        └── test_size = 0.3
      │
      ▼
Evaluasi Model
  ├── Accuracy Score
  ├── Classification Report (Precision, Recall, F1)
  ├── Confusion Matrix
  ├── ROC AUC Score
  └── ROC Curve
      │
      ▼
Analisis Feature Importance
      │
      ▼
Antarmuka Prediksi Manual (Input Pengguna)
```

---

## 🤖 Model

**Algoritma:** Random Forest Classifier

```python
from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier(
    n_estimators=100,
    max_features='sqrt',
    random_state=42
)
rf.fit(X_train, y_train)
```

### Mengapa Random Forest?
- Menangani fitur numerik dengan baik
- Lebih tahan terhadap overfitting dibanding single decision tree
- Menghasilkan skor kepentingan fitur (feature importance)
- Cocok untuk data sensor / IoT

---

## 📊 Metrik Evaluasi

| Metrik | Keterangan |
|--------|------------|
| **Accuracy** | Jumlah prediksi benar / total prediksi |
| **Precision** | TP / (TP + FP) — seberapa banyak prediksi positif yang benar |
| **Recall** | TP / (TP + FN) — seberapa banyak data positif yang berhasil dikenali |
| **F1-Score** | 2 × (Precision × Recall) / (Precision + Recall) |
| **ROC AUC** | Luas area di bawah kurva ROC — mengukur kemampuan pemisahan kelas |

### Visualisasi yang Dihasilkan
- 📊 Histogram semua fitur numerik
- 📦 Boxplot untuk deteksi outlier
- 🔥 Heatmap korelasi antar fitur
- 🟦 Confusion Matrix (ON/OFF)
- 📈 Grafik Aktual vs Prediksi (50 sampel pertama)
- 📉 ROC Curve
- 📊 Bar chart Feature Importance

---

## 🔍 Temuan Utama

> **Fitur `Time`** memiliki skor kepentingan tertinggi dalam model Random Forest. Tanpa fitur ini, akurasi model kemungkinan akan menurun secara signifikan.

---

## 💻 Cara Menjalankan

### 1. Install dependensi
```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

### 2. Siapkan dataset
Letakkan file `TARP.csv` di direktori yang sama dengan notebook.

### 3. Jalankan notebook
```bash
jupyter notebook SmartSystem.ipynb
```

### 4. Prediksi manual (input interaktif)
Di bagian akhir notebook, kamu dapat memasukkan nilai sensor secara manual untuk mendapatkan prediksi secara langsung:
```
Masukkan nilai untuk masing-masing fitur berikut:

Soil Moisture: 45.2
Temperature: 28.5
Soil Humidity: 60.1
...

=== HASIL PREDIKSI ===
Status prediksi: ON
```



---

## 🛠️ Teknologi yang Digunakan

| Library | Kegunaan |
|---------|----------|
| `pandas` | Manipulasi dan analisis data |
| `numpy` | Komputasi numerik |
| `matplotlib` | Visualisasi data |
| `seaborn` | Visualisasi statistik |
| `scikit-learn` | Machine learning (Random Forest, metrik evaluasi) |

---

## 📄 Lisensi

Proyek ini dibuat untuk keperluan akademis. Dataset bersumber dari Kaggle sesuai lisensi yang berlaku.
