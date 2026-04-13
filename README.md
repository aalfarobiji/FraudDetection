# 🔍 Credit Card Fraud Detection — Tim 2

> Proyek Machine Learning untuk mendeteksi transaksi kartu kredit yang bersifat fraudulent menggunakan dataset transaksi nyata.

---

## 📋 Deskripsi Proyek

Proyek ini bertujuan membangun model klasifikasi biner untuk mendeteksi apakah suatu transaksi kartu kredit merupakan **fraud (1)** atau **non-fraud (0)**. Dataset yang digunakan memiliki karakteristik **imbalanced** (data fraud jauh lebih sedikit dari non-fraud), sehingga diperlukan teknik khusus seperti **SMOTE** untuk menanganinya.

---

## 📁 Struktur Proyek

```
fraud-detection/
│
├── Fraud_Detection_Tim2_AR.ipynb   # Notebook utama
├── fraudTrain.csv                  # Dataset training
└── README.md
```

---

## 📊 Dataset

Dataset berisi transaksi kartu kredit dengan **22 kolom** fitur, antara lain:

| Kolom | Deskripsi |
|-------|-----------|
| `trans_date_trans_time` | Waktu transaksi |
| `cc_num` | Nomor kartu kredit |
| `merchant` | Nama merchant |
| `category` | Kategori transaksi |
| `amt` | Jumlah nominal transaksi |
| `lat` / `long` | Koordinat pemegang kartu |
| `merch_lat` / `merch_long` | Koordinat merchant |
| `city_pop` | Populasi kota |
| `dob` | Tanggal lahir pemegang kartu |
| `is_fraud` | **Target**: 0 = Non-Fraud, 1 = Fraud |

> ⚠️ Dataset ini bersifat **imbalanced** — jumlah transaksi fraud jauh lebih sedikit dibandingkan non-fraud.

---

## 🔧 Alur Kerja (Pipeline)

### 1. Import Library
- `numpy`, `pandas`, `matplotlib`, `seaborn`
- `scikit-learn`, `imbalanced-learn`

### 2. Load Dataset
- Membaca `fraudTrain.csv`
- Eksplorasi awal: shape, info, describe

### 3. Exploratory Data Analysis (EDA)
- Distribusi kelas fraud vs non-fraud (pie chart)
- Analisis fraud berdasarkan **tahun** (2019 & 2020)
- Distribusi fitur numerik & correlation matrix (heatmap)
- Fraud rate per **kategori transaksi**

### 4. Preprocessing Data
- **Drop kolom identifier**: `cc_num`, `first`, `last`, `trans_num`
- Cek & imputasi missing value (median untuk numerik, modus untuk kategorikal)
- Cek duplikasi data

### 5. Feature Engineering
Beberapa fitur baru yang dibuat:

| Fitur Baru | Sumber | Keterangan |
|---|---|---|
| `risk_category_of_hour` | `hour` | Jam 00–03 & 22–23 = risiko tinggi |
| `distance` | `lat`, `long`, `merch_lat`, `merch_long` | Haversine distance (km) |
| `distance_category` | `distance` | Kategori: Dekat (1), Sedang (2), Jauh (3) |
| `age` | `trans_date` - `dob` | Usia pemegang kartu (tahun) |
| `category_basedon_fraud` | `category` | Encoding berdasarkan fraud rate per kategori |

> 📐 **Haversine Formula** digunakan untuk menghitung jarak geografis antara lokasi pemegang kartu dan merchant.

### 6. Feature Selection
Kolom yang tidak digunakan dalam modelling di-drop, termasuk kolom datetime mentah, koordinat asli, dan fitur redundan.

### 7. Train-Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

### 8. Handling Imbalanced Data — SMOTE
```python
from imblearn.over_sampling import SMOTE
smote = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote.fit_resample(X_train, y_train)
```
> SMOTE hanya diterapkan pada **data training**, bukan data test.

### 9. Modelling

Tiga model yang digunakan dan dibandingkan:

| Model | Library |
|-------|---------|
| Logistic Regression | `sklearn.linear_model` |
| Random Forest | `sklearn.ensemble` |
| AdaBoost | `sklearn.ensemble` |

### 10. Evaluasi Model
- **Metrik utama**: Macro Average F1-Score
- **Visualisasi**: Confusion Matrix (heatmap)
- **Perbandingan** seluruh model dalam tabel ringkasan

### 11. Feature Importance
- Visualisasi feature importance dari model **Random Forest**

---

## 📈 Metrik Evaluasi

Model dievaluasi menggunakan **Macro Average F1-Score** untuk mengakomodasi ketidakseimbangan kelas:

```
Macro Avg F1 = rata-rata F1 dari setiap kelas (tanpa mempertimbangkan proporsi)
```

---

## 🛠️ Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn
```

Atau install langsung dari notebook:
```bash
pip install imbalanced-learn
```

---

## 🚀 Cara Menjalankan

1. Clone repositori ini:
   ```bash
   git clone https://github.com/<username>/fraud-detection.git
   cd fraud-detection
   ```

2. Pastikan file `fraudTrain.csv` sudah tersedia di direktori yang sama dengan notebook.

3. Jalankan notebook:
   ```bash
   jupyter notebook Fraud_Detection_Tim2_AR.ipynb
   ```

4. Jalankan seluruh sel secara berurutan dari atas ke bawah.

---

## 👥 Tim

**Tim 2 — AR**

---

## 📌 Catatan

- Model dilatih menggunakan **data original** (tanpa SMOTE) sebagai baseline.
- Hyperparameter tuning tersedia sebagai langkah opsional (sudah disiapkan namun di-comment).
- Untuk pengembangan lebih lanjut, dapat dicoba: XGBoost, LightGBM, atau deep learning approach.
