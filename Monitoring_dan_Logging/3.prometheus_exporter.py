import time
import json
from pathlib import Path

import psutil
from prometheus_client import Gauge, start_http_server

# 1. Lokasi experiment_summary.json
BASE_DIR = Path(__file__).resolve().parent
SUMMARY_PATH = BASE_DIR.parent / "Membangun_model" / "reports" / "experiment_summary.json"

# 2. Metrik Sistem (dinamis)
CPU_USAGE = Gauge("system_cpu_usage_percent", "CPU usage percentage")
MEMORY_USAGE = Gauge("system_memory_usage_percent", "Memory usage percentage")

# 3. 10 Test Metrics Model (dinamis, dibaca dari JSON)
TEST_ACCURACY = Gauge("model_test_accuracy", "Test accuracy of the model")
TEST_PRECISION = Gauge("model_test_precision", "Test precision of the model")
TEST_RECALL = Gauge("model_test_recall", "Test recall of the model")
TEST_F1 = Gauge("model_test_f1", "Test F1-score of the model")
TEST_ROC_AUC = Gauge("model_test_roc_auc", "Test ROC-AUC of the model")
TEST_SPECIFICITY = Gauge("model_test_specificity", "Test specificity (TN / (TN+FP))")
TEST_NPV = Gauge(
    "model_test_negative_predictive_value",
    "Test negative predictive value (TN / (TN+FN))",
)
TEST_FPR = Gauge("model_test_false_positive_rate", "Test false positive rate (FP / (FP+TN))")
TEST_FNR = Gauge("model_test_false_negative_rate", "Test false negative rate (FN / (FN+TP))")
TEST_BALANCED_ACC = Gauge(
    "model_test_balanced_accuracy", "Test balanced accuracy of the model"
)

# Turunan dari accuracy
ERROR_RATE = Gauge("model_test_error_rate", "1 - test accuracy")


def update_system_metrics():
    """Update metrik sistem dari psutil."""
    CPU_USAGE.set(psutil.cpu_percent())
    MEMORY_USAGE.set(psutil.virtual_memory().percent)


def update_model_test_metrics():
    """
    Baca 10 test metric dari experiment_summary.json.
    File ini diperbarui oleh Membangun_model/modelling_tuning.py
    (misalnya lewat cron job yang menjalankannya berulang).
    """
    try:
        with SUMMARY_PATH.open("r") as f:
            summary = json.load(f)
    except FileNotFoundError:
        print(f"[WARN] experiment_summary.json tidak ditemukan di {SUMMARY_PATH}")
        return
    except json.JSONDecodeError:
        print(f"[WARN] Gagal parse JSON dari {SUMMARY_PATH}")
        return

    test_metrics = summary.get("test_metrics", {})

    acc = float(test_metrics.get("accuracy", 0.0))
    prec = float(test_metrics.get("precision", 0.0))
    rec = float(test_metrics.get("recall", 0.0))
    f1 = float(test_metrics.get("f1", 0.0))
    roc_auc = float(test_metrics.get("roc_auc", 0.0))
    spec = float(test_metrics.get("specificity", 0.0))
    npv = float(test_metrics.get("negative_predictive_value", 0.0))
    fpr = float(test_metrics.get("false_positive_rate", 0.0))
    fnr = float(test_metrics.get("false_negative_rate", 0.0))
    bal_acc = float(test_metrics.get("balanced_accuracy", 0.0))

    TEST_ACCURACY.set(acc)
    TEST_PRECISION.set(prec)
    TEST_RECALL.set(rec)
    TEST_F1.set(f1)
    TEST_ROC_AUC.set(roc_auc)
    TEST_SPECIFICITY.set(spec)
    TEST_NPV.set(npv)
    TEST_FPR.set(fpr)
    TEST_FNR.set(fnr)
    TEST_BALANCED_ACC.set(bal_acc)
    ERROR_RATE.set(1.0 - acc)


def main():
    start_http_server(8000)
    print("Prometheus exporter berjalan di port 8000")
    print(f"Membaca test metric dari: {SUMMARY_PATH}")

    while True:
        update_system_metrics()
        update_model_test_metrics()
        time.sleep(5)  # update tiap 5 detik


if __name__ == "__main__":
    main()
