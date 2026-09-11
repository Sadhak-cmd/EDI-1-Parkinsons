"""
train_model.py
==============
Trains and evaluates two classifiers (Random Forest and SVM) on the
per-subject aggregated gait features produced by build_dataset.py.

Pipeline
--------
1. **GroupKFold cross-validation** (5 folds, grouped by subject_id) —
   selects the better classifier based on mean balanced accuracy.  No
   subject's data ever appears in both a training fold and its paired
   validation fold.
2. **GroupShuffleSplit hold-out** (25 % of subjects) — provides a single
   clean test set for the final evaluation metrics reported below.
3. **Winner fit** — the better classifier (from step 1) is re-fitted on
   the full training portion of the hold-out split and evaluated on the
   held-out test subjects.
4. **Model persistence** — the final fitted pipeline (scaler + classifier)
   is saved to ``models/gait_rf_model.pkl`` via joblib.

Key design choices
------------------
* **GroupKFold** (not plain StratifiedKFold) for CV — subjects are the
  grouping unit, preventing data leakage even for multi-trial subjects.
  Per-subject CSV already has one row per subject, so this is equivalent
  to leave-N-subjects-out.
* **class_weight='balanced'** on both classifiers — compensates for any
  residual PD / control imbalance without discarding samples.
* **Balanced accuracy** used to select the winner — equally sensitive to
  both classes, appropriate for the ~50/50 split.
* **StandardScaler** fit only on each training fold / training split and
  applied to the corresponding validation / test fold — no leakage.
* **SVC(probability=True)** — enables predict_proba for future ROC / AUC
  evaluation without re-fitting.

Input
-----
data/gait_features_by_subject.csv  (produced by build_dataset.py)

Output
------
models/gait_rf_model.pkl   (fitted pipeline: StandardScaler + winner clf)

Run from the project root:
    .venv/bin/python3 src/train_model.py
"""

import sys
import os

SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble        import RandomForestClassifier
from sklearn.svm             import SVC
from sklearn.calibration     import CalibratedClassifierCV
from sklearn.preprocessing   import StandardScaler
from sklearn.pipeline        import Pipeline
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics         import (
    balanced_accuracy_score,
    accuracy_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
)

# ── Configuration ─────────────────────────────────────────────────────────────

INPUT_CSV   = os.path.join(PROJECT_ROOT, "data", "gait_features_by_subject.csv")
MODEL_DIR   = os.path.join(PROJECT_ROOT, "models")
MODEL_PATH  = os.path.join(MODEL_DIR, "gait_rf_model.pkl")

FEATURE_COLS = [
    "cadence", "stride_time_mean", "stride_time_cv", "symmetry_index",
    "swing_stance_ratio", "double_support_time", "freeze_proxy",
    "step_force_asymmetry", "stride_regularity", "stride_interval_entropy",
]
LABEL_COL   = "label"
GROUP_COL   = "subject_id"

# GroupKFold CV parameters
CV_N_FOLDS  = 5

# GroupShuffleSplit hold-out parameters
TEST_SIZE   = 0.25
RANDOM_SEED = 42

# ── Classifier definitions ────────────────────────────────────────────────────

CLASSIFIERS: dict[str, object] = {
    "RandomForest": RandomForestClassifier(
        n_estimators=100,
        max_depth=6,
        class_weight="balanced",
        random_state=RANDOM_SEED,
        n_jobs=-1,
    ),
    "SVC": CalibratedClassifierCV(
        SVC(
            kernel="rbf",
            C=1.0,
            gamma="scale",
            class_weight="balanced",
            random_state=RANDOM_SEED,
        ),
        ensemble=False,   # single Platt-scaling fit; sklearn 1.9+ recommended
        cv=5,
    ),
}

# ── Helpers ───────────────────────────────────────────────────────────────────

def _make_pipeline(clf) -> Pipeline:
    """Wrap a classifier in a StandardScaler pipeline."""
    return Pipeline([
        ("scaler", StandardScaler()),
        ("clf",    clf),
    ])


def groupkfold_cv(
    clf,
    X: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
) -> dict:
    """Run GroupKFold CV and return per-fold and mean metrics.

    The scaler is fit inside each fold's training data only, preventing
    any leakage of test-fold statistics into the feature scaling.
    """
    gkf = GroupKFold(n_splits=CV_N_FOLDS)

    accs, bal_accs = [], []
    all_y_true, all_y_pred = [], []

    for fold, (train_idx, val_idx) in enumerate(
        gkf.split(X, y, groups=groups), start=1
    ):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]

        pipe = _make_pipeline(clf)
        pipe.fit(X_tr, y_tr)
        y_pred = pipe.predict(X_val)

        acc  = accuracy_score(y_val, y_pred)
        bacc = balanced_accuracy_score(y_val, y_pred)
        accs.append(acc)
        bal_accs.append(bacc)
        all_y_true.extend(y_val)
        all_y_pred.extend(y_pred)

        n_val_pd = int((y_val == 1).sum())
        n_val_co = int((y_val == 0).sum())
        print(f"    Fold {fold}:  acc={acc:.3f}  bal_acc={bacc:.3f}"
              f"  (val: {n_val_pd} PD, {n_val_co} Control)")

    return {
        "acc_mean":      float(np.mean(accs)),
        "acc_std":       float(np.std(accs)),
        "bal_acc_mean":  float(np.mean(bal_accs)),
        "bal_acc_std":   float(np.std(bal_accs)),
        "fold_accs":     accs,
        "fold_bal_accs": bal_accs,
        "all_y_true":    np.array(all_y_true),
        "all_y_pred":    np.array(all_y_pred),
    }


def print_held_out_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    clf_name: str,
) -> None:
    """Print the full evaluation table for the held-out test set."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc         = accuracy_score(y_true, y_pred)
    bal_acc     = balanced_accuracy_score(y_true, y_pred)
    precision   = precision_score(y_true, y_pred, pos_label=1, zero_division=0)
    recall      = recall_score(y_true, y_pred, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print(f"\n  Held-out test evaluation  ({clf_name})")
    print(f"  {'─' * 45}")
    print(f"  Accuracy              : {acc:.3f}")
    print(f"  Balanced accuracy     : {bal_acc:.3f}")
    print(f"  Precision  (PD class) : {precision:.3f}")
    print(f"  Recall / Sensitivity  : {recall:.3f}   (TP / (TP+FN))")
    print(f"  Specificity           : {specificity:.3f}   (TN / (TN+FP))")
    print(f"\n  Confusion matrix:")
    print(f"    {'':22s}  Predicted: 0  Predicted: 1")
    print(f"    {'Actual: 0 (Control)':22s}  {tn:^12d}  {fp:^12d}")
    print(f"    {'Actual: 1 (PD)':22s}  {fn:^12d}  {tp:^12d}")
    print(f"\n  Full classification report:")
    print(classification_report(
        y_true, y_pred,
        target_names=["Control (0)", "Parkinson's (1)"],
        digits=3,
    ))


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    # ── Load data ─────────────────────────────────────────────────────────────
    if not os.path.exists(INPUT_CSV):
        print(f"[ERROR] Input CSV not found:\n  {INPUT_CSV}")
        print("  Run  python src/build_dataset.py  first.")
        sys.exit(1)

    df = pd.read_csv(INPUT_CSV)
    print(f"Loaded {len(df)} subjects from {INPUT_CSV}")

    X      = df[FEATURE_COLS].to_numpy()
    y      = df[LABEL_COL].to_numpy()
    groups = df[GROUP_COL].to_numpy()

    n_pd = int((y == 1).sum())
    n_co = int((y == 0).sum())
    print(f"  Parkinson's (1): {n_pd}  |  Control (0): {n_co}")
    print(f"  Features ({len(FEATURE_COLS)}): {FEATURE_COLS}")
    print(f"\n  CV strategy  : GroupKFold (n_folds={CV_N_FOLDS}, grouped by {GROUP_COL})")
    print(f"  Hold-out     : GroupShuffleSplit (test_size={TEST_SIZE})\n")

    # ── Step 1: GroupKFold CV — select the better classifier ──────────────────
    cv_results: dict[str, dict] = {}

    for name, clf in CLASSIFIERS.items():
        print(f"{'─' * 55}")
        print(f"  GroupKFold CV — {name}")
        print(f"{'─' * 55}")

        res = groupkfold_cv(clf, X, y, groups)
        cv_results[name] = res

        print(f"\n  CV mean accuracy     : {res['acc_mean']:.3f}  (±{res['acc_std']:.3f})")
        print(f"  CV mean balanced acc : {res['bal_acc_mean']:.3f}  (±{res['bal_acc_std']:.3f})\n")

    # ── CV summary table ──────────────────────────────────────────────────────
    print(f"\n{'=' * 55}")
    print(f"  CV Summary  ({CV_N_FOLDS}-fold GroupKFold)")
    print(f"{'=' * 55}")
    print(f"  {'Classifier':<18}  {'Accuracy':>12}  {'Balanced Acc':>14}")
    print(f"  {'-'*18}  {'-'*12}  {'-'*14}")
    for name, res in cv_results.items():
        print(f"  {name:<18}  "
              f"{res['acc_mean']:.3f} ±{res['acc_std']:.3f}  "
              f"  {res['bal_acc_mean']:.3f} ±{res['bal_acc_std']:.3f}")
    print(f"{'=' * 55}\n")

    # ── Select winner by mean balanced accuracy ───────────────────────────────
    winner_name = max(cv_results, key=lambda n: cv_results[n]["bal_acc_mean"])
    winner_clf  = CLASSIFIERS[winner_name]
    margin = (cv_results[winner_name]["bal_acc_mean"] -
              cv_results[next(n for n in cv_results if n != winner_name)]["bal_acc_mean"])
    print(f"  Winner: {winner_name}  "
          f"(bal_acc={cv_results[winner_name]['bal_acc_mean']:.3f}, "
          f"margin={margin:+.3f})\n")

    # ── Step 2: GroupShuffleSplit — single held-out test split ────────────────
    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    n_test_pd = int((y_test == 1).sum())
    n_test_co = int((y_test == 0).sum())
    print(f"  Hold-out split — train: {len(train_idx)} subjects  "
          f"test: {len(test_idx)} subjects "
          f"(PD={n_test_pd}, Control={n_test_co})")

    # ── Step 3: Fit winner on train split, evaluate on test split ─────────────
    final_pipe = _make_pipeline(winner_clf)
    final_pipe.fit(X_train, y_train)
    y_pred_test = final_pipe.predict(X_test)

    print_held_out_metrics(y_test, y_pred_test, winner_name)

    # ── Step 4: Feature importances (Random Forest only) ─────────────────────
    if winner_name == "RandomForest":
        rf_clf      = final_pipe.named_steps["clf"]
        importances = rf_clf.feature_importances_
        order       = np.argsort(importances)[::-1]

        print(f"  Feature Importances  (mean decrease in impurity, held-out fit)")
        print(f"  {'─' * 50}")
        for rank, idx in enumerate(order, 1):
            bar = "█" * int(importances[idx] * 200)
            print(f"  {rank:2d}. {FEATURE_COLS[idx]:<28}  "
                  f"{importances[idx]:.4f}  {bar}")
        print()

    # ── Step 5: Save the final model ──────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(final_pipe, MODEL_PATH)
    print(f"  Model saved  → {MODEL_PATH}")
    print(f"  Pipeline     : StandardScaler → {winner_name}")
    print(f"  Reload with  : pipe = joblib.load('{MODEL_PATH}')")
    print("\nDone.")


if __name__ == "__main__":
    main()
