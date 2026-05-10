import pandas as pd
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    balanced_accuracy_score,
)
from pathlib import Path
import dagshub
import json
import os

# --- SETUP OTOMATIS ---
dagshub.init(repo_owner='farhanhanifazhary', repo_name='Heart-Failure', mlflow=True)

BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR.parent / "Membangun_model" / "heart_failure_clinical_records_dataset_preprocessing.csv"

# Load Data
df = pd.read_csv(DATASET_PATH)
X = df.drop('DEATH_EVENT', axis=1)
y = df['DEATH_EVENT']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- HYPERPARAMETER TUNING ---
print("Memulai Tuning...")

# Matikan autolog → semua harus manual
mlflow.sklearn.autolog(disable=True)

param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [10, 20],
    'min_samples_split': [2, 5]
}

rf = RandomForestClassifier(random_state=42)
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1)

with mlflow.start_run(run_name="RandomForest_GridSearch"):
    # ----------------- TRAINING & TUNING -----------------
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print(f"Best Params: {best_params}")
    print(f"Best CV Score: {best_score}")

    # ----------------- LOG PARAMS (manual) -----------------
    # Semua best params
    for param_name, value in best_params.items():
        mlflow.log_param(f"best_{param_name}", value)

    # Info grid / CV
    mlflow.log_param("cv_folds", grid_search.cv)
    mlflow.log_param("scoring", grid_search.scoring if grid_search.scoring is not None else "default_accuracy")
    mlflow.log_param("n_candidates", len(grid_search.cv_results_["params"]))

    # ----------------- LOG METRICS (manual) -----------------
    # CV metric utama
    mlflow.log_metric("best_cv_accuracy", best_score)

    # Hitung metric di test set pakai best model
    y_pred = best_model.predict(X_test)
    y_proba = best_model.predict_proba(X_test)[:, 1]

    # 5 metric test dasar
    test_accuracy = accuracy_score(y_test, y_pred)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    test_roc_auc = roc_auc_score(y_test, y_proba)

    # 5 metric tambahan dari confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    test_specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    test_npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    test_fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    test_fnr = fn / (fn + tp) if (fn + tp) > 0 else 0.0  # False Negative Rate
    test_balanced_acc = balanced_accuracy_score(y_test, y_pred)

    # Log 10 metric test ke MLflow
    mlflow.log_metric("test_accuracy", test_accuracy)
    mlflow.log_metric("test_precision", test_precision)
    mlflow.log_metric("test_recall", test_recall)
    mlflow.log_metric("test_f1", test_f1)
    mlflow.log_metric("test_roc_auc", test_roc_auc)
    mlflow.log_metric("test_specificity", test_specificity)
    mlflow.log_metric("test_negative_predictive_value", test_npv)
    mlflow.log_metric("test_false_positive_rate", test_fpr)
    mlflow.log_metric("test_false_negative_rate", test_fnr)
    mlflow.log_metric("test_balanced_accuracy", test_balanced_acc)

    # ----------------- LOG MODEL (manual, ganti autolog) -----------------
    mlflow.sklearn.log_model(
        sk_model=best_model,
        artifact_path="model_best_tuned",
        registered_model_name=None
    )

    # ----------------- ARTEFAK YANG SETARA AUTOLOG -----------------
    # Simpan cv_results_ sebagai CSV
    cv_results_df = pd.DataFrame(grid_search.cv_results_)
    cv_results_path = "cv_results.csv"
    cv_results_df.to_csv(cv_results_path, index=False)
    mlflow.log_artifact(cv_results_path, artifact_path="analysis")

    # Simpan feature importances sebagai CSV
    feature_importances = pd.DataFrame({
        "feature": X.columns,
        "importance": best_model.feature_importances_
    }).sort_values(by="importance", ascending=False)

    fi_path = "feature_importances.csv"
    feature_importances.to_csv(fi_path, index=False)
    mlflow.log_artifact(fi_path, artifact_path="analysis")

    # ----------------- +2 ARTEFAK TAMBAHAN (DI LUAR AUTOLOG) -----------------
    # Ringkasan experiment dalam JSON (isi diperluas 10 metric)
    summary = {
        "best_params": best_params,
        "best_cv_accuracy": best_score,
        "test_metrics": {
            "accuracy": test_accuracy,
            "precision": test_precision,
            "recall": test_recall,
            "f1": test_f1,
            "roc_auc": test_roc_auc,
            "specificity": test_specificity,
            "negative_predictive_value": test_npv,
            "false_positive_rate": test_fpr,
            "false_negative_rate": test_fnr,
            "balanced_accuracy": test_balanced_acc,
        }
    }

    os.makedirs("reports", exist_ok=True)

    EXPERIMENT_SUMMARY_PATH = BASE_DIR.parent / "Membangun_model" / "reports"
    FEATURE_USED_PATH = BASE_DIR.parent / "Membangun_model" / "reports"

    summary_path = os.path.join(EXPERIMENT_SUMMARY_PATH, "experiment_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    mlflow.log_artifact(summary_path, artifact_path="reports")

    # Artefak tambahan lain: daftar fitur yang dipakai
    features_path = os.path.join(FEATURE_USED_PATH, "features_used.txt")
    with open(features_path, "w") as f:
        f.write("\n".join(X.columns))
    mlflow.log_artifact(features_path, artifact_path="reports")

    print("Tuning Selesai! Semua metric & artefak dicatat manual.")