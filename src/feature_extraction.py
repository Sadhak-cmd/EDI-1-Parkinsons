"""
feature_extraction.py
=====================
Derives clinically-meaningful gait features from the peak indices returned
by ``preprocessing.segment_gait_cycles`` and, for the force-based features,
from the raw left/right force signals.

All timing features are computed purely from the timing between consecutive
force peaks, making them robust to sensor scaling differences across subjects
and recording sessions.  The force-amplitude and signal-structure features
additionally use the left and right total-force channels to capture bilateral
loading patterns and temporal regularity.

Typical usage
-------------
>>> from preprocessing import lowpass_filter, segment_gait_cycles
>>> from feature_extraction import compute_features
>>> import numpy as np
>>>
>>> data    = np.loadtxt("GaCo01_01.txt")
>>> sig_L   = lowpass_filter(data[:, 17])   # total_force_L
>>> sig_R   = lowpass_filter(data[:, 18])   # total_force_R
>>> peaks, _ = segment_gait_cycles(sig_L)
>>> feats   = compute_features(peaks, sig_L, sig_R)
>>> print(feats)
"""

import numpy as np


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _stance_threshold(signal: np.ndarray, fraction: float = 0.05) -> float:
    """Return a 'foot on ground' threshold relative to the signal range.

    A sample is considered to be in *stance* (foot loaded) when the force
    value exceeds ``min + fraction * (max - min)``.  The default 5 % of
    range is well above sensor noise but well below any genuine loading peak,
    matching the height_threshold used in ``segment_gait_cycles``.
    """
    return float(np.min(signal)) + fraction * float(np.ptp(signal))


def _stride_regularity(signal: np.ndarray, fs: float = 100.0) -> float:
    """Autocorrelation-based stride regularity index.

    Computes the normalized autocorrelation of *signal* and returns the
    height of the first dominant peak that falls within the physiologically
    plausible stride-period window [0.8 s, 2.5 s].

    The search window 0.8–2.5 s corresponds to stride rates of 0.4–1.25
    strides/s (24–75 strides/min), covering slow PD shuffling through brisk
    normal walking.  Anything outside this range is artefact or noise.

    Normalization: the autocorrelation is divided by its zero-lag value
    (total signal power) so the result is bounded in [-1, 1], with 1.0
    indicating perfect stride-to-stride regularity.

    Returns 0.0 if the signal is too short or constant.
    """
    n = len(signal)
    if n < 2:
        return 0.0
    # Zero-mean before autocorrelating to remove DC offset
    s = signal - signal.mean()
    power = float(np.dot(s, s))
    if power == 0.0:
        return 0.0

    # Full autocorrelation via direct dot-product for each lag of interest.
    # The stride window in samples at the given sampling rate:
    lag_min = int(0.8 * fs)   # 80 samples at 100 Hz
    lag_max = int(2.5 * fs)   # 250 samples at 100 Hz
    lag_max = min(lag_max, n - 1)

    if lag_min >= lag_max:
        return 0.0

    acf = np.array(
        [float(np.dot(s[:n - lag], s[lag:])) / power
         for lag in range(lag_min, lag_max + 1)]
    )
    return float(np.max(acf))   # peak height = regularity score


def _sample_entropy(x: np.ndarray, m: int = 2, r_scale: float = 0.2) -> float:
    """Sample entropy of a 1-D time series.

    Parameters
    ----------
    x : np.ndarray
        The time series (here: step-interval sequence, not raw force signal).
    m : int
        Template length.  ``m=2`` is the standard default from Richman &
        Moorman (2000, Am J Physiol Heart Circ Physiol) and is the most
        widely cited choice in gait analysis.
    r_scale : float
        Tolerance as a fraction of the time series standard deviation.
        ``r = 0.2 * std`` is the canonical default from Richman & Moorman;
        it was shown to be insensitive to sample size across N=30–300 and
        gives good reproducibility across healthy and pathological gait.

    Returns
    -------
    float
        Sample entropy value.  Higher values → more irregular (less
        predictable) gait timing, consistent with PD motor impairment.
        Returns ``np.nan`` if fewer than ``m+2`` data points are available
        or if no template matches are found (log of zero undefined).
    """
    N = len(x)
    if N < m + 2:
        return float("nan")

    std = float(np.std(x, ddof=1))
    if std == 0.0:
        return 0.0   # perfectly regular → zero entropy

    r = r_scale * std

    def _count_matches(template_len: int) -> int:
        """Count pairs of templates of length *template_len* within r."""
        count = 0
        for i in range(N - template_len):
            # Chebyshev distance: max of absolute element-wise differences
            diff = np.abs(x[i: i + template_len] -
                          x[i + 1: i + 1 + (N - template_len - i)].reshape(
                              N - template_len - i, template_len))
            # diff shape: (N-template_len-i, template_len)
            matches = np.all(diff <= r, axis=1)
            count += int(np.sum(matches))
        return count

    # Vectorised version: avoid the inner reshape trick and use broadcasting
    def _count_matches_fast(template_len: int) -> int:
        count = 0
        for i in range(N - template_len):
            template = x[i: i + template_len]
            # Compare with all subsequent windows (no self-match)
            for j in range(i + 1, N - template_len):
                if np.max(np.abs(template - x[j: j + template_len])) <= r:
                    count += 2   # symmetric: (i,j) and (j,i)
        return count

    A = _count_matches_fast(m + 1)
    B = _count_matches_fast(m)

    if B == 0 or A == 0:
        # Undefined: no template matches or no longer matches found.
        # Return a large finite value rather than inf/nan so it doesn't
        # break downstream scaling; 2.0 is above the practical range for
        # gait (~0–1.5).
        return 2.0

    return float(-np.log(A / B))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_features(
    peaks: np.ndarray,
    signal_L: np.ndarray,
    signal_R: np.ndarray,
    fs: float = 100.0,
) -> dict:
    """Compute gait regularity, symmetry, and bilateral loading features.

    Ten features are returned in total — four timing-based (original), four
    force-signal-based, and two signal-structure features (new):

    Timing features
    ---------------
    - **cadence**: Number of steps taken per minute; reduced cadence is a
      hallmark of Parkinson's hypokinesia and bradykinesia.
    - **stride_time_mean**: Average duration of one full stride (two
      consecutive steps); elevated or variable stride time reflects motor
      slowing or freezing episodes.
    - **stride_time_cv**: Coefficient of variation (std / mean) of step
      intervals; higher values indicate greater step-to-step variability,
      a sensitive biomarker of Parkinson's motor control impairment.
    - **symmetry_index**: Mean absolute difference between alternating step
      intervals (odd vs even steps) normalised by the mean step interval;
      values above ~5 % suggest left/right leg asymmetry.

    Force-signal features (new)
    ---------------------------
    - **swing_stance_ratio**: Per-gait-cycle ratio of swing phase duration
      (force near zero, foot off ground) to stance phase duration (force
      above threshold), averaged across all detected cycles in the recording.
      PD patients typically show a reduced swing phase relative to stance.
    - **double_support_time**: Proportion of the total recording duration
      where *both* left and right force signals simultaneously exceed their
      respective 'foot on ground' thresholds.  This bilateral contact phase
      increases with motor impairment as subjects adopt a more cautious,
      flat-footed gait.
    - **freeze_proxy**: Count of gait cycles within this recording where the
      stride duration falls more than 2 standard deviations below the mean
      stride time for this recording.  Short-stride outliers are a proxy for
      festination or micro-freezing episodes characteristic of PD.
    - **step_force_asymmetry**: Mean absolute difference between left and
      right peak force values per step, normalised by their sum.  Yields a
      dimensionless ratio in [0, 1]; higher values reflect greater loading
      asymmetry between limbs, common in unilateral or asymmetric PD.

    Signal-structure features (new)
    --------------------------------
    - **stride_regularity**: Normalized autocorrelation peak of the left
      force signal within the physiologically plausible stride-period window
      [0.8 s, 2.5 s].  A value near 1.0 indicates highly regular,
      metronome-like gait; lower values reflect stride-to-stride variability
      characteristic of PD.  Based on the Moe-Nilssen (2004) approach.
    - **stride_interval_entropy**: Sample entropy of the step-interval time
      series (``m=2``, ``r=0.2*std`` per Richman & Moorman 2000).  Higher
      values indicate less predictable (more irregular) step timing, a
      well-established PD biomarker.  Computed on step intervals, not the
      raw force signal, so it is independent of sensor amplitude.

    Parameters
    ----------
    peaks : np.ndarray, shape (P,)
        Integer sample indices of detected step peaks (from the LEFT force
        channel), as returned by ``preprocessing.segment_gait_cycles``.
        Must contain at least 4 peaks.
    signal_L : np.ndarray, shape (N,)
        Filtered total left-foot force signal (column 17 of the raw file).
    signal_R : np.ndarray, shape (N,)
        Filtered total right-foot force signal (column 18 of the raw file).
    fs : float, optional
        Sampling rate in **Hz**.  Default ``100.0``.

    Returns
    -------
    dict
        Keys: ``cadence``, ``stride_time_mean``, ``stride_time_cv``,
        ``symmetry_index``, ``swing_stance_ratio``, ``double_support_time``,
        ``freeze_proxy``, ``step_force_asymmetry``, ``stride_regularity``,
        ``stride_interval_entropy``.
        ``freeze_proxy`` is an ``int`` (raw cycle count); all others are
        ``float``.

    Raises
    ------
    ValueError
        If ``peaks`` has fewer than 4 elements.
    """
    if len(peaks) < 4:
        raise ValueError(
            f"At least 4 step peaks are required to compute features; "
            f"got {len(peaks)}.  Check that the signal is long enough and "
            f"that min_step_interval is not too large."
        )

    # ── Step intervals in seconds ────────────────────────────────────────────
    step_intervals: np.ndarray = np.diff(peaks) / fs   # shape (P-1,)

    # ── Cadence ─────────────────────────────────────────────────────────────
    cadence: float = 60.0 / float(np.mean(step_intervals))

    # ── Stride time (2 consecutive steps = 1 full gait cycle) ───────────────
    n_pairs = len(step_intervals) // 2
    stride_intervals: np.ndarray = (
        step_intervals[:2 * n_pairs:2] + step_intervals[1:2 * n_pairs:2]
    )   # shape (n_pairs,)

    stride_time_mean: float = float(np.mean(stride_intervals))

    # ── Step-time coefficient of variation ──────────────────────────────────
    stride_time_cv: float = float(
        np.std(step_intervals, ddof=1) / np.mean(step_intervals)
    )

    # ── Symmetry index ───────────────────────────────────────────────────────
    left_steps  = step_intervals[0::2]
    right_steps = step_intervals[1::2]
    min_len     = min(len(left_steps), len(right_steps))
    left_steps  = left_steps[:min_len]
    right_steps = right_steps[:min_len]
    mean_step   = float(np.mean(step_intervals))
    symmetry_index: float = float(
        np.mean(np.abs(left_steps - right_steps)) / mean_step
    )

    # ────────────────────────────────────────────────────────────────────────
    # Force-signal features
    # ────────────────────────────────────────────────────────────────────────

    thr_L = _stance_threshold(signal_L)
    thr_R = _stance_threshold(signal_R)

    # Boolean masks: True where the foot is in stance phase
    stance_L: np.ndarray = signal_L >= thr_L   # shape (N,)
    stance_R: np.ndarray = signal_R >= thr_R   # shape (N,)

    # ── 1. Swing / stance ratio ──────────────────────────────────────────────
    # For each detected gait cycle (between consecutive left-foot peaks),
    # compute the number of samples in swing vs stance for the LEFT foot,
    # then take the per-cycle ratio and average across all cycles.
    ratios: list[float] = []
    for i in range(len(peaks) - 1):
        cycle_slice = stance_L[peaks[i]: peaks[i + 1]]
        n_stance = int(np.sum(cycle_slice))
        n_swing  = len(cycle_slice) - n_stance
        if n_stance > 0:               # guard against degenerate cycles
            ratios.append(n_swing / n_stance)

    swing_stance_ratio: float = float(np.mean(ratios)) if ratios else 0.0

    # ── 2. Double-support time ───────────────────────────────────────────────
    # Proportion of samples where both feet are simultaneously in stance.
    # Computed over the full recording (not just cycle windows) so that
    # transitions between cycles are also captured.
    both_stance: np.ndarray = stance_L & stance_R
    double_support_time: float = float(np.mean(both_stance))

    # ── 3. Freeze proxy ──────────────────────────────────────────────────────
    # Count of gait cycles whose stride duration is more than 2 SD below the
    # mean stride time for this recording.  Uses stride_intervals (one entry
    # per paired left-step cycle) already computed above.
    if len(stride_intervals) >= 2:
        stride_mean      = float(np.mean(stride_intervals))
        stride_std       = float(np.std(stride_intervals, ddof=1))
        short_threshold  = stride_mean - 2.0 * stride_std
        freeze_proxy: int = int(np.sum(stride_intervals < short_threshold))
    else:
        freeze_proxy = 0

    # ── 4. Step force asymmetry ──────────────────────────────────────────────
    # For each cycle window (peak[i] → peak[i+1]):
    #   • f_L = left force at the left-foot peak sample
    #   • f_R = maximum right force within the same window (the R peak may
    #           not coincide exactly with the L peak, so we take the max)
    # Normalised absolute difference: |f_L - f_R| / (f_L + f_R)
    asym_values: list[float] = []
    for i in range(len(peaks) - 1):
        f_L   = float(signal_L[peaks[i]])
        f_R   = float(np.max(signal_R[peaks[i]: peaks[i + 1]]))
        denom = f_L + f_R
        if denom > 0:
            asym_values.append(abs(f_L - f_R) / denom)

    step_force_asymmetry: float = (
        float(np.mean(asym_values)) if asym_values else 0.0
    )

    # ── 5. Stride regularity ────────────────────────────────────────────────
    # Normalized autocorrelation peak in the stride-period window [0.8, 2.5] s.
    # Uses the filtered left-foot signal which has the clearest periodic
    # structure (peak detection was also done on this channel).
    stride_regularity: float = _stride_regularity(signal_L, fs=fs)

    # ── 6. Stride interval entropy ──────────────────────────────────────────
    # Sample entropy of the step-interval sequence.
    # Operating on step_intervals (not force signal) makes this feature
    # independent of absolute force magnitude and sensor calibration.
    raw_entropy = _sample_entropy(step_intervals, m=2, r_scale=0.2)
    # Replace nan with a neutral fallback (median typical gait SampEn ~0.5)
    stride_interval_entropy: float = (
        float(raw_entropy) if not np.isnan(raw_entropy) else 0.5
    )

    return {
        # ── Timing features (original) ───────────────────────────────────────
        "cadence":                  cadence,
        "stride_time_mean":         stride_time_mean,
        "stride_time_cv":           stride_time_cv,
        "symmetry_index":           symmetry_index,
        # ── Force-signal features ────────────────────────────────────────────
        "swing_stance_ratio":       swing_stance_ratio,
        "double_support_time":      double_support_time,
        "freeze_proxy":             freeze_proxy,
        "step_force_asymmetry":     step_force_asymmetry,
        # ── Signal-structure features (new) ──────────────────────────────────
        "stride_regularity":        stride_regularity,
        "stride_interval_entropy":  stride_interval_entropy,
    }
