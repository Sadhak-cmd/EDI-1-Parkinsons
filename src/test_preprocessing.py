"""
test_preprocessing.py
=====================
Smoke-test / visual QA script for src/preprocessing.py.

Loads the first file from the PhysioNet Gait-in-Parkinson's dataset,
applies the low-pass filter and gait-cycle segmentation, then produces
a three-panel diagnostic chart saved as output_preprocessing_check.png.

Run from the project root:
    python src/test_preprocessing.py

Requirements: numpy, scipy, matplotlib  (all available in the project venv)
"""

import sys
import os

# Allow running from any directory; resolve the src/ folder
SRC_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SRC_DIR)
sys.path.insert(0, SRC_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # non-interactive backend — no display needed
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from preprocessing import lowpass_filter, segment_gait_cycles

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

DATA_FILE = os.path.join(
    PROJECT_ROOT,
    "physionet.org", "files", "gaitpdb", "1.0.0", "GaCo01_01.txt"
)

# Column indices (0-based)
COL_TIME          = 0
COL_TOTAL_FORCE_L = 17   # total vertical force under LEFT foot

FS     = 100.0   # sampling rate (Hz)
CUTOFF = 10.0    # low-pass cutoff (Hz)
ORDER  = 4       # Butterworth order
MIN_STEP_INTERVAL = 0.4  # minimum inter-peak gap (s)

OUTPUT_PATH = os.path.join(PROJECT_ROOT, "output_preprocessing_check.png")

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
print(f"Loading: {DATA_FILE}")
data = np.loadtxt(DATA_FILE)

time          = data[:, COL_TIME]
raw_signal    = data[:, COL_TOTAL_FORCE_L]

print(f"  Samples loaded : {len(time):,}")
print(f"  Duration       : {time[-1]:.1f} s")
print(f"  Column used    : total_force_L  (col {COL_TOTAL_FORCE_L})")

# ---------------------------------------------------------------------------
# 2. Filter
# ---------------------------------------------------------------------------
print(f"\nApplying low-pass filter  (cutoff={CUTOFF} Hz, order={ORDER}) ...")
filtered_signal = lowpass_filter(raw_signal, cutoff=CUTOFF, fs=FS, order=ORDER)

# ---------------------------------------------------------------------------
# 3. Segment gait cycles
# ---------------------------------------------------------------------------
print(f"Segmenting gait cycles    (min_step_interval={MIN_STEP_INTERVAL} s) ...")
peaks, cycles = segment_gait_cycles(filtered_signal, fs=FS,
                                     min_step_interval=MIN_STEP_INTERVAL)

n_cycles = len(cycles)
print(f"\n{'=' * 50}")
print(f"  Gait cycles detected : {n_cycles}")
print(f"  Peak indices (first 5): {peaks[:5].tolist()}")
if n_cycles > 0:
    lengths = [len(c) for c in cycles]
    print(f"  Cycle length range   : {min(lengths)}–{max(lengths)} samples  "
          f"({min(lengths)/FS:.2f}–{max(lengths)/FS:.2f} s)")
print(f"{'=' * 50}\n")

# ---------------------------------------------------------------------------
# 4. Plot
# ---------------------------------------------------------------------------

# --- Style -----------------------------------------------------------------
plt.rcParams.update({
    "figure.facecolor":  "#0d1117",
    "axes.facecolor":    "#161b22",
    "axes.edgecolor":    "#30363d",
    "axes.labelcolor":   "#c9d1d9",
    "xtick.color":       "#8b949e",
    "ytick.color":       "#8b949e",
    "text.color":        "#c9d1d9",
    "grid.color":        "#21262d",
    "grid.linestyle":    "--",
    "grid.linewidth":    0.6,
    "legend.framealpha": 0.3,
    "legend.edgecolor":  "#30363d",
    "font.family":       "DejaVu Sans",
    "font.size":         10,
})

COLOUR_RAW      = "#58a6ff"   # blue
COLOUR_FILTERED = "#3fb950"   # green
COLOUR_PEAKS    = "#f78166"   # coral-red
COLOUR_SEGMENTS = "#a5d6ff"   # light blue (cycle shading)

fig = plt.figure(figsize=(18, 10))
fig.suptitle(
    "Gait Force Preprocessing Quality Check  ·  total_force_L  ·  GaCo01_01.txt",
    fontsize=14, fontweight="bold", color="#e6edf3", y=0.98
)

gs = gridspec.GridSpec(3, 1, figure=fig, hspace=0.55)

# ── Panel 1: Raw signal ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0])
ax1.plot(time, raw_signal, color=COLOUR_RAW, linewidth=0.7, alpha=0.9,
         label="Raw signal")
ax1.set_title("① Raw Signal", loc="left", fontsize=11, fontweight="bold",
               color="#e6edf3", pad=6)
ax1.set_ylabel("Force (N)")
ax1.legend(loc="upper right", fontsize=9)
ax1.grid(True)
ax1.set_xlim(time[0], time[-1])

# ── Panel 2: Raw + Filtered overlay ─────────────────────────────────────────
ax2 = fig.add_subplot(gs[1], sharex=ax1)
ax2.plot(time, raw_signal, color=COLOUR_RAW, linewidth=0.5, alpha=0.45,
         label="Raw signal")
ax2.plot(time, filtered_signal, color=COLOUR_FILTERED, linewidth=1.2,
         label=f"Filtered ({CUTOFF} Hz LP, order {ORDER})")
ax2.set_title("② Raw vs Filtered", loc="left", fontsize=11, fontweight="bold",
               color="#e6edf3", pad=6)
ax2.set_ylabel("Force (N)")
ax2.legend(loc="upper right", fontsize=9)
ax2.grid(True)

# ── Panel 3: Filtered + peaks + shaded cycles ────────────────────────────────
ax3 = fig.add_subplot(gs[2], sharex=ax1)
ax3.plot(time, filtered_signal, color=COLOUR_FILTERED, linewidth=1.1,
         label="Filtered signal")

# Shade alternating gait cycle bands
for i, cyc in enumerate(cycles):
    t_start = time[peaks[i]]
    t_end   = time[peaks[i + 1]]
    ax3.axvspan(t_start, t_end, alpha=0.08 if i % 2 == 0 else 0.0,
                color=COLOUR_SEGMENTS, linewidth=0)

# Mark the peaks
ax3.scatter(
    time[peaks], filtered_signal[peaks],
    color=COLOUR_PEAKS, s=40, zorder=5, linewidths=0.5,
    edgecolors="#ffffff", label=f"Step peaks  (n={len(peaks)})"
)

ax3.set_title("③ Detected Gait Cycles", loc="left", fontsize=11,
               fontweight="bold", color="#e6edf3", pad=6)
ax3.set_ylabel("Force (N)")
ax3.set_xlabel("Time (s)")
ax3.legend(loc="upper right", fontsize=9)
ax3.grid(True)

# Annotate cycle count in corner
ax3.text(
    0.01, 0.97,
    f"Cycles detected: {n_cycles}",
    transform=ax3.transAxes,
    va="top", ha="left", fontsize=10, fontweight="bold",
    color=COLOUR_PEAKS,
    bbox=dict(boxstyle="round,pad=0.3", facecolor="#21262d", edgecolor=COLOUR_PEAKS,
              linewidth=1.0, alpha=0.85)
)

plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight",
            facecolor=fig.get_facecolor())
plt.close(fig)

print(f"Chart saved  →  {OUTPUT_PATH}")
print("Done.")
