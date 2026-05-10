# Heart Failure Prediction: End-to-End MLOps Project
[![Docker](https://img.shields.io/badge/Docker-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Prometheus](https://img.shields.io/badge/Prometheus-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)](https://prometheus.io/)
[![Grafana](https://img.shields.io/badge/Grafana-F46800?style=for-the-badge&logo=grafana&logoColor=white)](https://grafana.com/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Heart--Failure-blue?logo=dagshub&logoColor=white)](https://dagshub.com/farhanhanifazhary/Heart-Failure)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)

Proyek ini merupakan implementasi sistem MLOps *end-to-end* untuk memprediksi risiko kegagalan jantung menggunakan dataset rekam medis klinis. Proyek ini mencakup seluruh siklus hidup pengembangan model, mulai dari eksperimen data, pelacakan model, otomasi CI/CD, hingga pemantauan performa di lingkungan produksi.

## 🚀 Fitur Utama

* **Eksperimen Terotomasi:** Alur kerja preprocessing data yang terdokumentasi dalam notebook Jupyter dan skrip otomasi.
* **Model Tracking & Registry:** Integrasi dengan **MLflow** dan **DagsHub** untuk mencatat parameter, metrik, dan artefak model secara *real-time*.
* **Continuous Integration (CI):** Pipeline **GitHub Actions** yang secara otomatis melatih model, membangun Docker Image, dan mengunggahnya ke **Docker Hub** setiap ada perubahan kode pada branch main.
* **Monitoring & Alerting:** Sistem pemantauan menggunakan **Prometheus** dan **Grafana** untuk melacak kesehatan sistem serta degradasi metrik performa model.

## 🛠️ Tech Stack

* **Bahasa:** Python 3.12.7
* **Machine Learning:** Scikit-Learn (Random Forest Classifier), Pandas, NumPy
* **MLOps & DevOps:** MLflow 2.19.0, DagsHub, Docker, GitHub Actions
* **Monitoring:** Prometheus, Grafana, Psutil

## 📂 Struktur Proyek

```text
├── Eksperimen/              # Notebook eksperimen dan skrip preprocessing
├── Membangun_model/         # Skrip training, tuning, dan manajemen MLflow
│   ├── modelling.py         # Skrip training utama dengan logging MLflow
│   └── requirements.txt     # Daftar dependensi proyek
├── Workflow-CI/             # Konfigurasi otomasi CI/CD
│   └── .github/workflows/   # Definisi pipeline GitHub Actions (ci.yml)
└── Monitoring_dan_Logging/  # Implementasi monitoring produksi
    ├── prometheus_exporter.py # Exporter metrik sistem dan model
    └── inference.py         # Simulasi layanan inferensi (serving)
```

## 📋 Komponen Proyek

### 1. Eksperimen & Preprocessing
Tahap awal proyek difokuskan pada analisis data dan pembersihan untuk memastikan kualitas input model:
* **Analisis Data:** Eksperimen dilakukan menggunakan Jupyter Notebook untuk eksplorasi fitur dan penanganan data yang tidak seimbang.
* **Otomasi Preprocessing:** Skrip Python digunakan untuk melakukan transformasi data secara otomatis guna menghasilkan dataset siap latih (`heart_failure_clinical_records_dataset_preprocessing.csv`).

### 2. Membangun Model (Model Development)
Proyek ini mengimplementasikan model klasifikasi untuk memprediksi risiko kematian pasien:
* **Algoritma:** Menggunakan **Random Forest Classifier** dengan parameter `n_estimators=100` dan `max_depth=10`.
* **Pelacakan Eksperimen:** Integrasi penuh dengan **MLflow** dan **DagsHub** untuk mencatat parameter, durasi pelatihan, dan artefak model.
* **Metrik Evaluasi:** Model dievaluasi menggunakan 10 metrik utama: Accuracy, Precision, Recall, F1-Score, ROC-AUC, Specificity, Negative Predictive Value (NPV), False Positive Rate (FPR), False Negative Rate (FNR), dan Balanced Accuracy.

### 3. Workflow CI (Continuous Integration)
Sistem otomasi dibangun menggunakan GitHub Actions untuk memastikan integritas kode dan model:
* **Automated Training:** Setiap push ke branch `main` memicu proses pelatihan model otomatis di lingkungan virtual.
* **Dockerization:** Pipeline secara otomatis membangun Docker Image menggunakan perintah `mlflow models build-docker` setelah model berhasil dilatih.
* **Image Registry:** Image yang telah dibangun secara otomatis diunggah ke **Docker Hub** untuk kebutuhan deployment di tahap selanjutnya.

### 4. Monitoring & Logging
Implementasi pemantauan pasca-deployment untuk menjaga performa model di lingkungan produksi:
* **Prometheus Exporter:** Skrip khusus yang mengekspor metrik sistem (penggunaan CPU dan Memori) serta metrik performa model (Akurasi, Error Rate, dll.) dari file `experiment_summary.json`.
* **Inference Service Simulator:** Simulasi layanan prediksi yang menerima data pasien dummy dan menghasilkan log output prediksi secara real-time.
* **Visualisasi Metrik:** Data dari Prometheus dapat dihubungkan ke **Grafana** untuk visualisasi metrik performa dan pengaturan sistem peringatan (alerting).

## 👨‍💻 Kontributor
Farhan Hanif Azhary - Cloud & ML Engineer student
