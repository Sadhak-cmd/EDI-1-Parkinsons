"""
train_tflite_model.py
=====================
Phase 6: Trains a lightweight neural network for mobile deployment and converts
it to a standalone TensorFlow Lite (.tflite) model for Parkinson's Disease gait screening.

Key features:
1. In-model Normalization: Extracts exact mean and variance vectors from the existing
   StandardScaler in models/gait_rf_model.pkl and builds them into a Keras Normalization
   layer. This guarantees that the exported .tflite model directly accepts RAW gait
   feature values in their original units.
2. Identical Hold-out Split: Uses GroupShuffleSplit(test_size=0.25, random_state=42)
   grouped by subject_id, strictly matching the baseline split.
3. Mobile Optimization: Compact architecture (< 500 parameters) suitable for real-time
   on-device inference via Flutter (tflite_flutter).
4. Direct TFLite Verification: Loads the exported .tflite file, runs a sample raw feature
   vector, and asserts numerical equivalence with the Keras model.

Outputs:
- models/gait_tflite_model.keras
- models/gait_model.tflite
"""

import os
import sys

# Suppress verbose TensorFlow logs and configure non-interactive cache
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["MPLCONFIGDIR"] = "/tmp/matplotlib"

SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)

sys.path.insert(0, SRC_DIR)

import joblib
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
from sklearn.utils.class_weight import compute_class_weight

import tensorflow as tf

# Set random seeds for reproducibility
tf.keras.utils.set_random_seed(42)

# ---------------------------------------------------------------------------
# Configuration & File Paths
# ---------------------------------------------------------------------------

DATA_PATH = os.path.join(PROJECT_ROOT, "data", "gait_features_by_subject.csv")
SCALER_MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "gait_rf_model.pkl")
KERAS_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "gait_tflite_model.keras")
TFLITE_OUTPUT_PATH = os.path.join(PROJECT_ROOT, "models", "gait_model.tflite")

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


def load_dataset_and_split():
    """Load subject-level dataset and apply the exact existing GroupShuffleSplit."""
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATA_PATH}. Run build_dataset.py first.")

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_COLS].to_numpy(dtype=np.float32)
    y = df[LABEL_COL].to_numpy(dtype=np.float32)
    groups = df[GROUP_COL].to_numpy()

    gss = GroupShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    train_idx, test_idx = next(gss.split(X, y, groups=groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    return X_train, X_test, y_train, y_test, df.iloc[test_idx]


def extract_scaler_parameters():
    """Load existing model pipeline and extract scaler statistics."""
    if not os.path.exists(SCALER_MODEL_PATH):
        raise FileNotFoundError(f"Existing model not found at {SCALER_MODEL_PATH}")

    pipeline = joblib.load(SCALER_MODEL_PATH)
    scaler = pipeline.named_steps["scaler"]

    mean = scaler.mean_.astype(np.float32)
    variance = scaler.var_.astype(np.float32)

    return mean, variance


def build_model(mean_vector: np.ndarray, var_vector: np.ndarray) -> tf.keras.Model:
    """Build a compact neural network with built-in feature standardization."""
    # Pre-configured normalization layer with exact training mean and variance
    norm_layer = tf.keras.layers.Normalization(
        axis=-1,
        mean=mean_vector,
        variance=var_vector,
        name="feature_normalization",
    )

    inputs = tf.keras.Input(shape=(len(FEATURE_COLS),), dtype=tf.float32, name="raw_gait_features")
    x = norm_layer(inputs)
    x = tf.keras.layers.Dense(
        16,
        activation="relu",
        kernel_regularizer=tf.keras.regularizers.l2(0.005),
        name="dense_1",
    )(x)
    x = tf.keras.layers.Dropout(0.2, name="dropout_1")(x)
    x = tf.keras.layers.Dense(8, activation="relu", name="dense_2")(x)
    outputs = tf.keras.layers.Dense(1, activation="sigmoid", name="pd_probability")(x)

    model = tf.keras.Model(inputs=inputs, outputs=outputs, name="gait_pd_screening_net")
    return model


def main():
    print("=" * 70)
    print("  Phase 6: TensorFlow Lite Model Training & Conversion")
    print("=" * 70)

    # 1. Load data
    X_train, X_test, y_train, y_test, test_df = load_dataset_and_split()
    print(f"\n[1] Dataset loaded from: {DATA_PATH}")
    print(f"    Train subjects : {len(X_train)}  (PD: {int((y_train == 1).sum())}, Control: {int((y_train == 0).sum())})")
    print(f"    Test subjects  : {len(X_test)}  (PD: {int((y_test == 1).sum())}, Control: {int((y_test == 0).sum())})")
    print(f"    Input features : {len(FEATURE_COLS)} (Order strictly preserved)")

    # 2. Extract scaler statistics
    mean_vec, var_vec = extract_scaler_parameters()
    print(f"\n[2] Scaler statistics extracted from: {SCALER_MODEL_PATH}")
    for i, col in enumerate(FEATURE_COLS):
        print(f"    {i+1:2d}. {col:<26s}  mean={mean_vec[i]:10.5f}  std={np.sqrt(var_vec[i]):10.5f}")

    # 3. Build model
    model = build_model(mean_vec, var_vec)
    print("\n[3] Model Architecture:")
    model.summary()

    # 4. Class weighting & Compilation
    classes = np.array([0, 1])
    class_weights_arr = compute_class_weight("balanced", classes=classes, y=y_train)
    class_weight_dict = {0: float(class_weights_arr[0]), 1: float(class_weights_arr[1])}
    print(f"\n[4] Class weights for training: {class_weight_dict}")

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.003)
    model.compile(
        optimizer=optimizer,
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    # 5. Train model
    print("\n[5] Training Keras model (60 epochs)...")
    history = model.fit(
        X_train,
        y_train,
        epochs=60,
        batch_size=8,
        validation_data=(X_test, y_test),
        class_weight=class_weight_dict,
        verbose=0,
    )

    final_train_acc = history.history["accuracy"][-1]
    final_val_acc = history.history["val_accuracy"][-1]
    print(f"    Training completed. Final Epoch Train Acc: {final_train_acc:.3f}, Val Acc: {final_val_acc:.3f}")

    # 6. Evaluate Keras Model on Held-out Test Set
    y_prob_test = model.predict(X_test, verbose=0).flatten()
    y_pred_test = (y_prob_test >= 0.5).astype(int)

    cm = confusion_matrix(y_test, y_pred_test)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, y_pred_test)
    bal_acc = balanced_accuracy_score(y_test, y_pred_test)
    precision = precision_score(y_test, y_pred_test, pos_label=1, zero_division=0)
    sensitivity = recall_score(y_test, y_pred_test, pos_label=1, zero_division=0)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    print("\n" + "─" * 60)
    print("  Held-out Test Evaluation (Keras Model)")
    print("─" * 60)
    print(f"  Accuracy              : {acc:.3f}")
    print(f"  Balanced Accuracy     : {bal_acc:.3f}")
    print(f"  Precision (PD class)  : {precision:.3f}")
    print(f"  Sensitivity / Recall  : {sensitivity:.3f}  (TP: {tp} / {tp+fn})")
    print(f"  Specificity (Control) : {specificity:.3f}  (TN: {tn} / {tn+fp})")
    print("\n  Confusion Matrix:")
    print(f"    {'':22s}  Predicted: 0 (Control)  Predicted: 1 (PD)")
    print(f"    {'Actual: 0 (Control)':22s}  {tn:^22d}  {fp:^18d}")
    print(f"    {'Actual: 1 (PD)':22s}  {fn:^22d}  {tp:^18d}")

    # 7. Save Keras Model
    os.makedirs(os.path.dirname(KERAS_OUTPUT_PATH), exist_ok=True)
    model.save(KERAS_OUTPUT_PATH)
    print(f"\n[6] Keras model saved → {KERAS_OUTPUT_PATH}")

    # 8. Convert to TensorFlow Lite (.tflite)
    print("\n[7] Converting to TensorFlow Lite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    with open(TFLITE_OUTPUT_PATH, "wb") as f:
        f.write(tflite_model)

    tflite_size_kb = os.path.getsize(TFLITE_OUTPUT_PATH) / 1024.0
    print(f"    TFLite model saved → {TFLITE_OUTPUT_PATH} ({tflite_size_kb:.1f} KB)")

    # 9. Verify TFLite Model with Interpreter
    print("\n[8] Verifying .tflite model with TensorFlow Lite Interpreter...")
    interpreter = tf.lite.Interpreter(model_path=TFLITE_OUTPUT_PATH)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    input_shape = input_details[0]["shape"]
    input_dtype = input_details[0]["dtype"]
    output_shape = output_details[0]["shape"]
    output_dtype = output_details[0]["dtype"]

    print(f"    Input Name   : '{input_details[0]['name']}'")
    print(f"    Input Shape  : {input_shape} (dtype: {input_dtype.__name__})")
    print(f"    Output Name  : '{output_details[0]['name']}'")
    print(f"    Output Shape : {output_shape} (dtype: {output_dtype.__name__})")

    # Run real test sample (sample index 0)
    sample_raw = X_test[0:1]  # Raw unscaled features
    true_label = int(y_test[0])
    sample_subject = test_df.iloc[0]["subject_id"]

    # Keras prediction
    keras_prob = float(model.predict(sample_raw, verbose=0)[0][0])

    # TFLite prediction
    interpreter.set_tensor(input_details[0]["index"], sample_raw)
    interpreter.invoke()
    tflite_prob = float(interpreter.get_tensor(output_details[0]["index"])[0][0])

    diff = abs(keras_prob - tflite_prob)
    print(f"\n    Sample Verification (Subject: {sample_subject}, True Label: {true_label}):")
    print(f"    - Raw Features: {sample_raw[0].round(3).tolist()}")
    print(f"    - Keras Output  : {keras_prob:.6f}")
    print(f"    - TFLite Output : {tflite_prob:.6f}")
    print(f"    - Abs Difference: {diff:.2e}")

    assert diff < 1e-4, f"Discrepancy too large between Keras and TFLite: {diff}"
    print("    => Direct TFLite verification PASSED successfully!")

    print("\n" + "=" * 70)
    print("  Phase 6 Complete: TensorFlow Lite Model Ready for Flutter App")
    print("=" * 70)


if __name__ == "__main__":
    main()
