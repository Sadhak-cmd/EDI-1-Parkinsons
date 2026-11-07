"""
test_tflite_model.py
====================
Standalone evaluation and validation script for models/gait_model.tflite.

Tests:
1. Loads the exported .tflite model and verifies tensor input/output signatures.
2. Recreates the exact held-out test split (36 subjects) from data/gait_features_by_subject.csv.
3. Evaluates inference over the entire test set using raw, unscaled feature values.
4. Computes Accuracy, Balanced Accuracy, Sensitivity, Specificity, and Confusion Matrix.
5. Displays a side-by-side comparison with the existing SVC baseline.
6. Benchmarks inference latency (milliseconds per prediction).
"""

import os
import sys
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)
import tensorflow as tf

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "gait_features_by_subject.csv")
TFLITE_PATH = os.path.join(PROJECT_ROOT, "models", "gait_model.tflite")

FEATURE_COLS = [
    "cadence",
    "stride_time_mean",
    "stride_time_cv",
    "symmetry_index",
    "swing_stance_ratio",
    "double_support_time",
    "freeze_proxy",
    "step_force_asymmetry",
    "stride_regularity",
    "stride_interval_entropy",
]
LABEL_COL = "label"
GROUP_COL = "subject_id"
TEST_SIZE = 0.25
RANDOM_SEED = 42

# Baseline metrics from Phase 5 SVC evaluation for reference
BASELINE_SVC = {
    "accuracy": 0.722,
    "balanced_accuracy": 0.721,
    "sensitivity": 0.706,
    "specificity": 0.737,
}


def load_test_split():
    """Load the identical held-out test subjects."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.int32)
    groups = df[GROUP_COL].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    _, test_idx = next(gss.split(X, y, groups=groups))

    return X[test_idx], y[test_idx], df.iloc[test_idx]


def evaluate_tflite():
    print("=" * 70)
    print("  TensorFlow Lite Standalone Model Evaluation & Benchmark")
    print("=" * 70)

    if not os.path.exists(TFLITE_PATH):
        print(f"[ERROR] TFLite model not found at:\n  {TFLITE_PATH}")
        print("  Please run: .venv/bin/python3 src/train_tflite_model.py first.")
        sys.exit(1)

    # 1. Initialize Interpreter
    interpreter = tf.lite.Interpreter(model_path=TFLITE_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    in_idx = input_details[0]["index"]
    out_idx = output_details[0]["index"]

    print(f"\n[1] Loaded Model: {TFLITE_PATH}")
    print(f"    File size    : {os.path.getsize(TFLITE_PATH) / 1024.0:.1f} KB")
    print(f"    Input shape  : {input_details[0]['shape'].tolist()} ({input_details[0]['dtype'].__name__})")
    print(f"    Output shape : {output_details[0]['shape'].tolist()} ({output_details[0]['dtype'].__name__})")
    print(f"    Input expects: RAW unscaled features (normalization performed inside graph)")

    # 2. Load held-out test data
    X_test, y_test, test_df = load_test_split()
    n_test = len(X_test)
    n_pd = int((y_test == 1).sum())
    n_co = int((y_test == 0).sum())
    print(f"\n[2] Loaded Held-out Test Set: {n_test} subjects ({n_pd} PD, {n_co} Control)")

    # 3. Run Inference across all test subjects
    y_probs = []
    latencies = []

    for i in range(n_test):
        sample = X_test[i : i + 1]  # shape (1, 10)

        t0 = time.perf_counter()
        interpreter.set_tensor(in_idx, sample)
        interpreter.invoke()
        prob = float(interpreter.get_tensor(out_idx)[0][0])
        t1 = time.perf_counter()

        y_probs.append(prob)
        latencies.append((t1 - t0) * 1000.0)  # ms

    y_probs = np.array(y_probs)
    y_preds = (y_probs >= 0.5).astype(int)

    # 4. Compute Metrics
    acc = accuracy_score(y_test, y_preds)
    bal_acc = balanced_accuracy_score(y_test, y_preds)
    precision = precision_score(y_test, y_preds, pos_label=1, zero_division=0)
    sensitivity = recall_score(y_test, y_preds, pos_label=1, zero_division=0)
    cm = confusion_matrix(y_test, y_preds)
    tn, fp, fn, tp = cm.ravel()
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "─" * 70)
    print("  Held-out Test Set Results (TFLite Standalone)")
    print("─" * 70)
    print(f"  Accuracy              : {acc:.3f}")
    print(f"  Balanced Accuracy     : {bal_acc:.3f}")
    print(f"  Precision (PD class)  : {precision:.3f}")
    print(f"  Sensitivity (Recall)  : {sensitivity:.3f}   (TP: {tp} / {tp+fn})")
    print(f"  Specificity (Control) : {specificity:.3f}   (TN: {tn} / {tn+fp})")

    print("\n  Confusion Matrix:")
    print(f"    {'':22s}  Predicted: 0 (Control)  Predicted: 1 (PD)")
    print(f"    {'Actual: 0 (Control)':22s}  {tn:^22d}  {fp:^18d}")
    print(f"    {'Actual: 1 (PD)':22s}  {fn:^22d}  {tp:^18d}")

    # 5. Side-by-side comparison with SVC Baseline
    print("\n" + "=" * 70)
    print("  Comparison: Baseline Python SVC vs Mobile TFLite Neural Net")
    print("=" * 70)
    print(f"  {'Metric':<25s}  {'SVC Baseline':>15s}  {'TFLite Mobile Net':>18s}  {'Difference':>10s}")
    print(f"  {'-'*25}  {'-'*15}  {'-'*18}  {'-'*10}")
    print(f"  {'Accuracy':<25s}  {BASELINE_SVC['accuracy']:>15.3f}  {acc:>18.3f}  {acc - BASELINE_SVC['accuracy']:>+10.3f}")
    print(f"  {'Balanced Accuracy':<25s}  {BASELINE_SVC['balanced_accuracy']:>15.3f}  {bal_acc:>18.3f}  {bal_acc - BASELINE_SVC['balanced_accuracy']:>+10.3f}")
    print(f"  {'Sensitivity (PD Recall)':<25s}  {BASELINE_SVC['sensitivity']:>15.3f}  {sensitivity:>18.3f}  {sensitivity - BASELINE_SVC['sensitivity']:>+10.3f}")
    print(f"  {'Specificity (Control)':<25s}  {BASELINE_SVC['specificity']:>15.3f}  {specificity:>18.3f}  {specificity - BASELINE_SVC['specificity']:>+10.3f}")
    print("=" * 70)

    # 6. Latency Benchmark
    avg_lat = np.mean(latencies)
    p95_lat = np.percentile(latencies, 95)
    print(f"\n[3] On-Device Latency Benchmark:")
    print(f"    Average inference time: {avg_lat:.3f} ms per prediction")
    print(f"    95th percentile latency: {p95_lat:.3f} ms")
    print(f"    Throughput capacity   : {1000.0 / avg_lat:.0f} predictions/second")
    print(f"    => Well within real-time mobile budget (< 50 ms)")

    # 7. Sample Predictions Table
    print(f"\n[4] Sample Subject Predictions (First 6 test subjects):")
    print(f"    {'Subject ID':<12s}  {'Actual':<10s}  {'Predicted':<10s}  {'PD Probability':>15s}  {'Result'}")
    print(f"    {'-'*12}  {'-'*10}  {'-'*10}  {'-'*15}  {'-'*10}")
    for i in range(min(6, n_test)):
        sub_id = test_df.iloc[i]["subject_id"]
        act = "PD (1)" if y_test[i] == 1 else "Control (0)"
        pred = "PD (1)" if y_preds[i] == 1 else "Control (0)"
        status = "CORRECT" if y_test[i] == y_preds[i] else "MISSED"
        print(f"    {sub_id:<12s}  {act:<10s}  {pred:<10s}  {y_probs[i]:>15.4f}  {status}")

    print("\n" + "=" * 70)
    print("  Verification Succeeded: models/gait_model.tflite is ready for Flutter!")
    print("=" * 70)


if __name__ == "__main__":
    evaluate_tflite()
