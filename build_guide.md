# Smartphone-Based Gait Analysis for Early PD Screening — Build Guide

## 1. Environment Setup

**Where to work:** Google Colab (colab.research.google.com) for everything in Phases 2–6. It gives you a free GPU/CPU, pre-installed Python, and nothing to configure locally.

Create one Colab notebook per phase to keep things organized:
- `01_data_exploration.ipynb`
- `02_preprocessing.ipynb`
- `03_feature_extraction.ipynb`
- `04_model_training.ipynb`
- `05_tflite_conversion.ipynb`

Install libraries at the top of each notebook:
```python
!pip install pandas numpy scipy scikit-learn tensorflow matplotlib
```

---

## 2. Get the Data

**Primary dataset: PhysioNet "Gait in Parkinson's Disease"**
- URL: physionet.org/content/gaitpdb/1.0.0/
- Contains vertical ground reaction force (VGRF) from 16 sensors under each foot, sampled at 100 Hz, for PD patients and healthy controls, plus a `demographics.xls` with labels (Hoehn & Yahr stage, control vs. PD).
- Download directly in Colab:
```python
!wget -r -N -c -np https://physionet.org/files/gaitpdb/1.0.0/
```
This isn't raw phone accelerometer data (it's force plates), but it's well-labeled and good for building your pipeline logic. Use it as your primary training set.

**Secondary dataset: mPower mobile study**
- Actual smartphone accelerometer/gyroscope walk-test data with PD labels.
- Requires free Synapse account (synapse.org) and accepting a data use agreement (automated, no waiting period for the walking subset).
- This is what makes your "smartphone-based" claim credible — use it for validation/testing, or as a second training source if you have time.

**Load and inspect:**
```python
import pandas as pd
df = pd.read_csv('path_to_file.txt', sep='\t', header=None)
df.columns = ['time'] + [f'sensor_{i}' for i in range(16)] + ['total_force_L', 'total_force_R']
print(df.head())
print(df.shape)
```

---

## 3. Preprocessing Pipeline

Build this as a reusable Python module, `preprocessing.py`, not inline notebook code — you'll import it later for the app logic too.

**Step A — Low-pass filter (remove sensor noise):**
```python
from scipy.signal import butter, filtfilt

def lowpass_filter(signal, cutoff=10, fs=100, order=4):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    return filtfilt(b, a, signal)
```

**Step B — Orientation correction (for phone-based accelerometer data):**
If you use mPower data, the phone can be in any orientation. Compute the magnitude vector to make it orientation-independent:
```python
import numpy as np

def acceleration_magnitude(ax, ay, az):
    return np.sqrt(ax**2 + ay**2 + az**2)
```
This sidesteps needing a full quaternion-based orientation correction — it's a standard simplification used in gait research and is defensible in your report.

**Step C — Gait cycle segmentation (peak detection):**
```python
from scipy.signal import find_peaks

def segment_gait_cycles(signal, fs=100, min_step_interval=0.4):
    distance = int(min_step_interval * fs)
    peaks, _ = find_peaks(signal, distance=distance, prominence=0.1)
    cycles = [signal[peaks[i]:peaks[i+1]] for i in range(len(peaks)-1)]
    return peaks, cycles
```
`min_step_interval=0.4` seconds prevents detecting noise as steps (normal cadence is roughly 100–120 steps/min → ~0.5–0.6s per step).

---

## 4. Feature Extraction

Build `feature_extraction.py`:

```python
import numpy as np

def compute_features(peaks, fs=100):
    step_times = np.diff(peaks) / fs  # seconds between steps

    cadence = 60 / np.mean(step_times)  # steps per minute
    stride_time_mean = np.mean(step_times) * 2  # one stride = two steps
    stride_time_std = np.std(step_times)
    stride_time_cv = stride_time_std / np.mean(step_times)  # variability, key PD marker

    # Symmetry: compare alternating step intervals (left vs right proxy)
    left_steps = step_times[0::2]
    right_steps = step_times[1::2]
    min_len = min(len(left_steps), len(right_steps))
    symmetry_index = np.abs(np.mean(left_steps[:min_len]) - np.mean(right_steps[:min_len]))

    return {
        'cadence': cadence,
        'stride_time_mean': stride_time_mean,
        'stride_time_cv': stride_time_cv,
        'symmetry_index': symmetry_index
    }
```

Run this over every subject's file, and build a feature table:
```python
rows = []
for file, label in zip(file_list, label_list):
    signal = load_and_preprocess(file)
    peaks, cycles = segment_gait_cycles(signal)
    features = compute_features(peaks)
    features['label'] = label
    rows.append(features)

feature_df = pd.DataFrame(rows)
feature_df.to_csv('gait_features.csv', index=False)
```
This CSV is your ML-ready dataset — save it, you'll reload it in the next notebook.

---

## 5. Train the Classifier

```python
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix

df = pd.read_csv('gait_features.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)

# Cross-validation to check stability on a small dataset
cv = StratifiedKFold(n_splits=5)
scores = cross_val_score(rf, X_train, y_train, cv=cv, scoring='accuracy')
print("CV accuracy:", scores.mean(), "+/-", scores.std())

rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))
```

Report **sensitivity** (recall on the PD class) explicitly — for a screening tool this matters more than raw accuracy, since missing a case is worse than a false alarm. State this reasoning in your report.

Try SVM (`SVC(kernel='rbf', probability=True)`) as a comparison model and report both.

Save the trained model:
```python
import joblib
joblib.dump(rf, 'gait_rf_model.pkl')
```

---

## 6. Convert to TensorFlow Lite

scikit-learn models don't convert directly to TFLite — you need a TensorFlow model. Two options:

**Option A (recommended for your timeline):** Re-implement a small feedforward neural net in Keras trained on the same feature table, matching the RF's performance as closely as possible, then convert that.

```python
import tensorflow as tf
from tensorflow import keras

model = keras.Sequential([
    keras.layers.Input(shape=(X_train.shape[1],)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, validation_split=0.2)

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()
with open('gait_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

**Option B (stretch goal):** Build the 1D-CNN directly on raw windowed signal instead of hand-crafted features, then convert that model the same way. Only attempt this after Option A works end-to-end.

Test the `.tflite` file standalone before touching Flutter:
```python
interpreter = tf.lite.Interpreter(model_path='gait_model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

interpreter.set_tensor(input_details[0]['index'], X_test.iloc[[0]].values.astype('float32'))
interpreter.invoke()
print(interpreter.get_tensor(output_details[0]['index']))
```

---

## 7. Build the Flutter App

**Setup:** Install Android Studio (free) + Flutter SDK. Create a new project:
```bash
flutter create gait_screen_app
cd gait_screen_app
```

**Add dependencies** in `pubspec.yaml`:
```yaml
dependencies:
  flutter:
    sdk: flutter
  sensors_plus: ^4.0.2
  tflite_flutter: ^0.10.4
```

**App structure (3 screens):**
```
lib/
  main.dart
  screens/
    home_screen.dart       # instructions + start button
    walk_test_screen.dart  # captures sensor data for 20-30 steps
    result_screen.dart     # shows "Normal" or "Consult a specialist"
  services/
    sensor_service.dart    # wraps sensors_plus, collects accel/gyro stream
    feature_extractor.dart # Dart port of your Python feature_extraction.py
    model_service.dart     # loads .tflite and runs inference
```

**Capturing sensor data** (`sensor_service.dart`):
```dart
import 'package:sensors_plus/sensors_plus.dart';

class SensorService {
  List<double> accelMagnitudes = [];

  void startlistening() {
    accelerometerEvents.listen((event) {
      double mag = (event.x * event.x + event.y * event.y + event.z * event.z);
      accelMagnitudes.add(mag);
    });
  }
}
```

**Running inference** (`model_service.dart`):
```dart
import 'package:tflite_flutter/tflite_flutter.dart';

class ModelService {
  late Interpreter interpreter;

  Future<void> loadModel() async {
    interpreter = await Interpreter.fromAsset('gait_model.tflite');
  }

  double predict(List<double> features) {
    var input = [features];
    var output = List.filled(1, 0.0).reshape([1, 1]);
    interpreter.run(input, output);
    return output[0][0];
  }
}
```

Put `gait_model.tflite` in `assets/` and register it in `pubspec.yaml` under `flutter: assets:`.

**Port your feature extraction logic** (peak detection, cadence, stride variability, symmetry) from Python to Dart — same math, just Dart syntax. This is the most error-prone step; test it against known Python outputs on the same sample signal to confirm they match before trusting it in the app.

---

## 8. Wire It Together

Flow: `walk_test_screen` collects ~20-30s of accelerometer data → pass to `feature_extractor.dart` → pass resulting feature vector to `model_service.dart` → display result on `result_screen.dart` with wording like:

> "This is a screening indicator, not a diagnosis. Please consult a neurologist for clinical evaluation."

Always keep this disclaimer visible on the result screen — it's both ethically necessary and something evaluators will specifically check for.

---

## 9. Validate and Report

- Re-run your held-out test set through the full pipeline (preprocessing → features → model) and confirm the reported accuracy/sensitivity/specificity match what you got in Colab.
- Document dataset sizes, class balance, and cross-validation results.
- List limitations explicitly: small sample size, force-plate data as a proxy for phone accelerometer, pocket-placement assumption, not a clinical diagnostic tool.
- Package into your final report (Word doc) and prepare a live demo recording as backup in case live sensor demo fails during evaluation.

---

## Suggested Timeline
| Phase | Time |
|---|---|
| Data setup + exploration | 3-4 days |
| Preprocessing + feature extraction | 5-7 days |
| Model training + evaluation | 4-5 days |
| TFLite conversion | 1-2 days |
| Flutter app + integration | 10-14 days |
| Validation + report writing | 4-5 days |
