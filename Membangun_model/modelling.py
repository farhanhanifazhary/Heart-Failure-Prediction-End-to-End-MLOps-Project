import pandas as pd
import time
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    roc_auc_score,
    balanced_accuracy_score,
)
import dagshub

# --- SETUP DAGSHUB OTOMATIS ---
dagshub.init(repo_owner='farhanhanifazhary', repo_name='Heart-Failure', mlflow=True)

# --- 2. LOAD DATA ---
df = pd.read_csv('heart_failure_clinical_records_dataset_preprocessing.csv')

X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']

# Split Data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- TRAINING DENGAN LOGGING ADVANCE ---
print("Memulai Training...")

# Matikan autolog (sesuai kode asli kamu – tetap dibiarkan)
mlflow.sklearn.autolog()

with mlflow.start_run(run_name="RandomForest_Autolog"):
    
    # 1. Training & Hitung Waktu (Manual Metric 1)
    start_time = time.time()
    model = RandomForestClassifier(n_estimators=100, max_depth=10, random_state=42)
    model.fit(X_train, y_train)
    end_time = time.time()
    duration = end_time - start_time
    
    # 2. Evaluasi Tambahan
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Confusion matrix untuk berbagai metric
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # 10 test metric (sama dengan di tuning)
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba)
    test_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred)

    print(f"Training Duration: {duration}s")
    print(f"Specificity: {specificity}")

    print("Selesai!")