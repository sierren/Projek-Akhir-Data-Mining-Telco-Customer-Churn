# Projek Akhir Data Mining - Telco Customer Churn

## Deskripsi Proyek

Proyek ini merupakan tugas akhir mata kuliah Data Mining yang bertujuan untuk melakukan prediksi customer churn dan segmentasi pelanggan pada industri telekomunikasi menggunakan dataset Telco Customer Churn.

Penelitian dilakukan menggunakan pendekatan CRISP-DM yang mencakup tahapan Business Understanding, Data Understanding, Data Preparation, Modeling, Evaluation, dan Deployment.

Model klasifikasi dibangun menggunakan algoritma XGBoost untuk memprediksi kemungkinan pelanggan melakukan churn. Selain itu, dilakukan segmentasi pelanggan menggunakan algoritma K-Means Clustering untuk mengelompokkan pelanggan berdasarkan karakteristik yang dimiliki.

Hasil analisis kemudian diimplementasikan ke dalam aplikasi web berbasis Streamlit agar dapat digunakan secara interaktif dan mudah dipahami.

---

## Tujuan Proyek

- Memprediksi pelanggan yang berpotensi melakukan churn.
- Mengidentifikasi faktor-faktor yang mempengaruhi churn pelanggan.
- Melakukan segmentasi pelanggan berdasarkan karakteristik yang dimiliki.
- Mengimplementasikan hasil analisis ke dalam aplikasi web interaktif.

---

## Dataset

Dataset yang digunakan adalah **Telco Customer Churn Dataset**.

Jumlah data:
- 7.043 pelanggan

Jumlah atribut:
- 21 atribut utama sebelum preprocessing

Contoh atribut:
- Gender
- SeniorCitizen
- Partner
- Dependents
- Tenure
- InternetService
- Contract
- PaymentMethod
- MonthlyCharges
- TotalCharges
- Churn

---

## Metode yang Digunakan

### 1. XGBoost
Digunakan untuk memprediksi customer churn berdasarkan karakteristik pelanggan.

### 2. K-Means Clustering
Digunakan untuk melakukan segmentasi pelanggan ke dalam beberapa kelompok berdasarkan kemiripan karakteristik.

### 3. SMOTE
Digunakan pada tahap preprocessing untuk mengatasi ketidakseimbangan data churn dan non-churn.

### 4. SHAP
Digunakan untuk menjelaskan kontribusi setiap fitur terhadap hasil prediksi model.

---

## Hasil Evaluasi

### XGBoost

| Metrik | Nilai |
|---------|---------|
| Accuracy | 76.65% |
| Precision | 55.27% |
| Recall | 63.10% |
| ROC-AUC | 82.56% |

### K-Means Clustering

| Metrik | Nilai |
|---------|---------|
| Jumlah Cluster | 3 |
| Silhouette Score | 0.2193 |

---

## Fitur Aplikasi

Aplikasi web dibangun menggunakan Streamlit dan memiliki fitur:

### Home
- Informasi proyek
- Ringkasan hasil analisis
- Identitas pengembang

### Dataset Overview
- Informasi dataset
- Statistik sederhana
- Visualisasi data

### Prediction / Analysis
- Form input pelanggan
- Prediksi customer churn
- Tingkat risiko churn

### Visualization
- Distribusi churn
- Analisis fitur
- Visualisasi clustering

### About
- Penjelasan metode
- Informasi dataset
- Informasi proyek

---

## Struktur Repository

```text
dataset/
model/
notebook/
streamlit_app/
laporan/

app.py
style.css
requirements.txt
README.md
```

---

## Menjalankan Aplikasi

Install library yang dibutuhkan:

```bash
pip install -r requirements.txt
```

Jalankan aplikasi:

```bash
streamlit run app.py
```

---

## Pengembang

**Sierren Yorensa**  
NIM: 24051214173

Program Studi Sistem Informasi

Universitas Negeri Surabaya

---

## Mata Kuliah

Data Mining
