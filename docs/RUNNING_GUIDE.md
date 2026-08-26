# Panduan Menjalankan Project

## Prasyarat

- Python 3.10 atau lebih tinggi
- pip (package manager)
- Git
- [Kaggle account](https://www.kaggle.com/) untuk download dataset (opsional, dataset sudah tersedia di `data/raw/`)

---

## 1. Clone Repository

```bash
git clone https://github.com/Bagus2510/fake-news-detection.git
cd fake-news-detection
```

---

## 2. Setup Virtual Environment

### Windows
```bash
python -m venv venv
venv\Scripts\activate
```

### macOS/Linux
```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 4. Dataset

Dataset ISOT Fake News Detection sudah tersedia di folder `data/raw/`:
- `Fake.csv` — 23,481 artikel fake news
- `True.csv` — 21,417 artikel real news

Jika ingin download ulang dari Kaggle:
1. Buka [Kaggle - Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
2. Download file `Fake.csv` dan `True.csv`
3. Pindahkan ke folder `data/raw/`

Struktur folder setelah download:
```
data/
├── raw/
│   ├── Fake.csv              # 23,481 artikel fake
│   └── True.csv              # 21,417 artikel real
├── processed/
│   └── (akan diisi otomatis setelah preprocessing)
```

---

## 5. Jalankan Notebook

### Urutan notebook (wajib berurutan):

| No | Notebook | Fungsi | Estimasi Waktu |
|----|----------|--------|----------------|
| 1 | `01_eda.ipynb` | Exploratory Data Analysis | ~2 menit |
| 2 | `02_preprocessing.ipynb` | Text Preprocessing | ~5 menit |
| 3 | `03_feature_extraction.ipynb` | TF-IDF Feature Extraction | ~3 menit |
| 4 | `04_modeling.ipynb` | Model Training & Evaluation | ~15 menit |

### Cara menjalankan:

```bash
cd notebooks
jupyter notebook
```

Klik notebook sesuai urutan, lalu jalankan semua cell (`Kernel` → `Restart & Run All`).

### Urutan penting:
- **01_eda** → Menghasilkan `data/processed/raw_combined.csv`
- **02_preprocessing** → Menghasilkan `data/processed/cleaned_data.csv`
- **03_feature_extraction** → Menghasilkan `data/processed/X_train_tfidf.npz`, `X_test_tfidf.npz`, `y_train.npy`, `y_test.npy`
- **04_modeling** → Menghasilkan `models/model_best.pkl` dan visualisasi di `images/`

> **Catatan**: Jika mengubah preprocessing di notebook 02, harus run ulang notebook 03 dan 04.

---

## 6. Jalankan Streamlit Demo

```bash
cd streamlit
pip install -r requirements.txt
streamlit run app.py
```

Aplikasi akan terbuka di browser. Fitur:
- **Predict tab** — Input teks atau pilih sample, lihat hasil klasifikasi + confidence score
- **EDA Insights tab** — Visualisasi dari notebook
- **About tab** — Ringkasan project

---

## Struktur Project

```
Fake News Detection/
├── data/
│   ├── raw/                    # Dataset asli (Fake.csv, True.csv)
│   └── processed/              # Hasil preprocessing
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_feature_extraction.ipynb
│   └── 04_modeling.ipynb
├── models/                     # Model yang sudah di-save
│   ├── model_best.pkl
│   └── tfidf_vectorizer.pkl
├── images/                     # Visualisasi dari notebook
├── streamlit/                  # Streamlit demo app
│   ├── app.py
│   ├── requirements.txt
│   └── .streamlit/config.toml
├── docs/                       # Dokumentasi
│   └── RUNNING_GUIDE.md
├── requirements.txt
└── README.md
```

---

## Troubleshooting

### Error: `ModuleNotFoundError`
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Error: `FileNotFoundError: data/raw/Fake.csv`
- Pastikan file `Fake.csv` dan `True.csv` ada di folder `data/raw/`

### Error: `FileNotFoundError: data/processed/X_train_tfidf.npz`
- Pastikan notebook `03_feature_extraction.ipynb` sudah dijalankan sebelum `04_modeling.ipynb`

### Model tidak bisa load
- Pastikan notebook `04_modeling.ipynb` sudah dijalankan
- File `model_best.pkl` harus ada di folder `models/`

---

## Hasil yang Diharapkan

Setelah menjalankan semua notebook:

1. **01_eda.ipynb** → Visualisasi distribusi label, subject leakage, text length, top words
2. **02_preprocessing.ipynb** → File `data/processed/cleaned_data.csv` (44,689 baris)
3. **03_feature_extraction.ipynb** → TF-IDF matrix (50,000 fitur), split train/test
4. **04_modeling.ipynb** → Model terbaik: LinearSVC (99.62% accuracy), 9 visualisasi di `images/`

---

## Catatan

- Dataset bersumber dari [Fake and Real News Dataset](https://www.kaggle.com/datasets/clmentbisaillon/fake-and-real-news-dataset)
- `random_state=43` digunakan di semua splitting & model untuk reproducibility
- **Subject column dihapus** karena tidak ada overlap antara kelas (data leakage)
- **Reuters attribution patterns** dihapus secara agresif karena muncul di 23% real news vs 1.4% fake news
- **TF-IDF di-fit hanya pada data training** untuk mencegah data leakage dari test set
