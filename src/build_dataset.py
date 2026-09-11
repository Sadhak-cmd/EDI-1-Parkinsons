"""
build_dataset.py
================
Pipeline script: loops over every .txt file in the PhysioNet Gait-in-
Parkinson's-Disease dataset, applies preprocessing + feature extraction, and
saves a tidy CSV ready for model training.

Output
------
data/gait_features.csv
    One row per successfully processed file.
    Columns: filename, subject_id, label, cadence, stride_time_mean,
             stride_time_cv, symmetry_index, swing_stance_ratio,
             double_support_time, freeze_proxy, step_force_asymmetry,
             stride_regularity, stride_interval_entropy

data/gait_features_by_subject.csv
    One row per unique subject (trials averaged / summed).
    Columns: subject_id, label, cadence, stride_time_mean,
             stride_time_cv, symmetry_index, swing_stance_ratio,
             double_support_time, freeze_proxy, step_force_asymmetry,
             stride_regularity, stride_interval_entropy
    Note: freeze_proxy is *summed* across a subject's trials; all other
    features are averaged.  This is the file used for model training.

Labels
------
Filename prefix determines the diagnostic label:
  *Pt* (e.g. GaPt, JuPt, SiPt) -> 1  (Parkinson's Disease)
  *Co* (e.g. GaCo, JuCo, SiCo) -> 0  (healthy Control)
  Any other prefix               -> skipped (label unknown)

Run from the project root:
    python src/build_dataset.py
"""

import sys
import os
import glob
import traceback

import numpy as np
import pandas as pd

# ── Resolve paths regardless of cwd ─────────────────────────────────────────
SRC_DIR      = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

from preprocessing     import lowpass_filter, segment_gait_cycles
from feature_extraction import compute_features

# ── Configuration ────────────────────────────────────────────────────────────
DATA_DIR = os.path.join(
    PROJECT_ROOT, "physionet.org", "files", "gaitpdb", "1.0.0"
)
OUTPUT_CSV         = os.path.join(PROJECT_ROOT, "data", "gait_features.csv")
OUTPUT_SUBJECT_CSV = os.path.join(PROJECT_ROOT, "data", "gait_features_by_subject.csv")

# Column indices for total force under LEFT and RIGHT feet (0-based)
COL_TOTAL_FORCE_L = 17
COL_TOTAL_FORCE_R = 18

# Preprocessing / segmentation parameters
FS               = 100.0   # Hz
CUTOFF           = 10.0    # Hz  low-pass cutoff
FILTER_ORDER     = 4       # Butterworth order
MIN_STEP_INTERVAL = 0.4   # seconds

# Minimum usable peaks; files with fewer are skipped rather than producing
# noisy / degenerate features
MIN_PEAKS = 10


# ── Filename helpers ─────────────────────────────────────────────────────────

def get_label(filename: str) -> int | None:
    """Return 1 (PD), 0 (control), or None if prefix is unrecognised."""
    base = os.path.basename(filename)
    # Prefixes: Ga/Ju/Si followed by Co or Pt
    if "Pt" in base[:6]:
        return 1
    if "Co" in base[:6]:
        return 0
    return None


def get_subject_id(filename: str) -> str:
    """Extract the subject identifier from a filename.

    Examples
    --------
    ``GaPt01_01.txt`` -> ``GaPt01``
    ``JuCo03_02.txt`` -> ``JuCo03``

    The convention is: everything before the first underscore.
    """
    base = os.path.basename(filename)
    return base.split("_")[0]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def process_file(filepath: str) -> dict:
    """Load one file, run the full pipeline, return a feature dict.

    Uses pandas to load the TSV so that rows with an unexpected number of
    columns (e.g. truncated lines in SiPt18_01.txt) are silently dropped
    rather than raising an exception.
    """
    # on_bad_lines='skip' discards any row that doesn't have exactly the
    # expected number of columns; usecols ensures we only read cols 0-18.
    df = pd.read_csv(
        filepath,
        sep="\t",
        header=None,
        usecols=range(19),
        on_bad_lines="skip",
        dtype=float,
        engine="python",
    )
    signal_L = df.iloc[:, COL_TOTAL_FORCE_L].to_numpy()
    signal_R = df.iloc[:, COL_TOTAL_FORCE_R].to_numpy()

    filtered_L       = lowpass_filter(signal_L, cutoff=CUTOFF,
                                      fs=FS, order=FILTER_ORDER)
    filtered_R       = lowpass_filter(signal_R, cutoff=CUTOFF,
                                      fs=FS, order=FILTER_ORDER)
    peaks, _         = segment_gait_cycles(filtered_L, fs=FS,
                                           min_step_interval=MIN_STEP_INTERVAL)

    if len(peaks) < MIN_PEAKS:
        raise ValueError(
            f"Only {len(peaks)} peaks detected (minimum required: {MIN_PEAKS})."
        )

    feats = compute_features(peaks, filtered_L, filtered_R, fs=FS)
    return feats


def main() -> None:
    txt_files = sorted(glob.glob(os.path.join(DATA_DIR, "*.txt")))

    if not txt_files:
        print(f"[ERROR] No .txt files found in:\n  {DATA_DIR}")
        sys.exit(1)

    print(f"Found {len(txt_files)} .txt files in dataset directory.")
    print(f"Processing …\n")

    rows: list[dict]  = []
    failed: list[str] = []

    for filepath in txt_files:
        basename = os.path.basename(filepath)

        # ── Determine label ──────────────────────────────────────────────────
        label = get_label(basename)
        if label is None:
            print(f"  [SKIP]  {basename}  — unrecognised prefix (not Pt/Co)")
            continue

        # ── Process with fault isolation ─────────────────────────────────────
        try:
            feats = process_file(filepath)
        except Exception as exc:
            reason = f"{type(exc).__name__}: {exc}"
            print(f"  [FAIL]  {basename}  — {reason}")
            failed.append(f"{basename}: {reason}")
            continue

        rows.append({
            "filename":                basename,
            "subject_id":              get_subject_id(basename),
            "label":                   label,
            "cadence":                 feats["cadence"],
            "stride_time_mean":        feats["stride_time_mean"],
            "stride_time_cv":          feats["stride_time_cv"],
            "symmetry_index":          feats["symmetry_index"],
            "swing_stance_ratio":      feats["swing_stance_ratio"],
            "double_support_time":     feats["double_support_time"],
            "freeze_proxy":            feats["freeze_proxy"],
            "step_force_asymmetry":    feats["step_force_asymmetry"],
            "stride_regularity":       feats["stride_regularity"],
            "stride_interval_entropy": feats["stride_interval_entropy"],
        })
        print(f"  [OK]    {basename}  "
              f"label={label}  "
              f"cadence={feats['cadence']:.1f} spm  "
              f"cv={feats['stride_time_cv']:.4f}  "
              f"dbl_sup={feats['double_support_time']:.3f}")

    # ── Assemble per-file DataFrame ──────────────────────────────────────────
    df = pd.DataFrame(rows, columns=[
        "filename", "subject_id", "label",
        "cadence", "stride_time_mean", "stride_time_cv", "symmetry_index",
        "swing_stance_ratio", "double_support_time", "freeze_proxy",
        "step_force_asymmetry", "stride_regularity", "stride_interval_entropy",
    ])

    os.makedirs(os.path.dirname(OUTPUT_CSV), exist_ok=True)
    df.to_csv(OUTPUT_CSV, index=False, float_format="%.6f")

    # ── Per-subject aggregation (mean across trials, sum for counts) ──────────
    # Each subject may have multiple recording sessions (_01, _02, …).
    # Continuous features are averaged; freeze_proxy is a cycle *count* and
    # is summed so that subjects with more recordings are not penalised.
    MEAN_COLS = [
        "cadence", "stride_time_mean", "stride_time_cv", "symmetry_index",
        "swing_stance_ratio", "double_support_time", "step_force_asymmetry",
        "stride_regularity", "stride_interval_entropy",
    ]
    df_subject = (
        df.groupby("subject_id", sort=True)
          .agg(
              label=pd.NamedAgg(column="label", aggfunc="first"),
              **{col: pd.NamedAgg(column=col, aggfunc="mean")
                 for col in MEAN_COLS},
              freeze_proxy=pd.NamedAgg(column="freeze_proxy", aggfunc="sum"),
          )
          .reset_index()
    )
    df_subject.to_csv(OUTPUT_SUBJECT_CSV, index=False, float_format="%.6f")

    # ── Summary ──────────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"  Successfully processed : {len(rows)} files")
    print(f"  Failed / skipped       : {len(failed)}")
    print(f"  Per-file CSV           : {OUTPUT_CSV}")
    print(f"  Per-subject CSV        : {OUTPUT_SUBJECT_CSV}")
    print("=" * 60)

    if df.shape[0] > 0:
        # Per-file breakdown
        n_pd_files = int((df["label"] == 1).sum())
        n_co_files = int((df["label"] == 0).sum())
        print(f"\n  File-level class breakdown:")
        print(f"    Parkinson's (label=1) : {n_pd_files}")
        print(f"    Control     (label=0) : {n_co_files}")

        # Per-subject breakdown
        n_pd_subj = int((df_subject["label"] == 1).sum())
        n_co_subj = int((df_subject["label"] == 0).sum())
        print(f"\n  Subject-level class breakdown ({len(df_subject)} unique subjects):")
        print(f"    Parkinson's (label=1) : {n_pd_subj}")
        print(f"    Control     (label=0) : {n_co_subj}")

        ALL_FEAT_COLS = [
            "cadence", "stride_time_mean", "stride_time_cv", "symmetry_index",
            "swing_stance_ratio", "double_support_time", "freeze_proxy",
            "step_force_asymmetry", "stride_regularity", "stride_interval_entropy",
        ]
        print(f"\n  Feature statistics (per-subject aggregates):")
        print(df_subject[ALL_FEAT_COLS].describe().round(4).to_string())

    if failed:
        print(f"\n  Files that failed:")
        for f in failed:
            print(f"    • {f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
