"""
preprocessing.py
================
Signal processing utilities for Gait-in-Parkinson's-Disease force data.

Each raw .txt file contains 19 tab-separated columns at 100 Hz:
  col 0        : time (s)
  cols 1–16    : individual vertical force sensors (8 left, 8 right)
  col 17       : total_force_L  — sum of all left-foot sensors  (N)
  col 18       : total_force_R  — sum of all right-foot sensors (N)

Typical usage
-------------
>>> import numpy as np
>>> from preprocessing import lowpass_filter, segment_gait_cycles
>>>
>>> data = np.loadtxt("GaCo01_01.txt")
>>> raw = data[:, 17]                          # total_force_L
>>> filtered = lowpass_filter(raw)
>>> peaks, cycles = segment_gait_cycles(filtered)
>>> print(f"{len(cycles)} gait cycles detected")
"""

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks


# ---------------------------------------------------------------------------
# 1. Low-pass Butterworth filter
# ---------------------------------------------------------------------------

def lowpass_filter(
    signal: np.ndarray,
    cutoff: float = 10.0,
    fs: float = 100.0,
    order: int = 4,
) -> np.ndarray:
    """Apply a zero-phase Butterworth low-pass filter to a 1-D signal.

    Uses ``scipy.signal.filtfilt`` (forward + backward pass) so that the
    group delay is exactly zero — no time-shift is introduced.  This
    preserves the temporal alignment of gait events (heel-strike, toe-off)
    while attenuating high-frequency electrical / mechanical noise above
    ``cutoff`` Hz.

    Why 10 Hz?
    ----------
    Human gait fundamental frequency sits between ~0.5–3 Hz; the first few
    harmonics that carry shape information are below 6 Hz.  A 10 Hz cutoff
    gives ≈ one octave of margin above the signal of interest, comfortably
    removing sensor noise (typically > 20 Hz) without blurring step peaks.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Raw 1-D force signal (any units; filter is unit-agnostic).
    cutoff : float, optional
        -3 dB frequency of the filter in **Hz**.  Default ``10.0``.
    fs : float, optional
        Sampling rate of the signal in **Hz**.  Must match the hardware
        acquisition rate.  Default ``100.0`` (PhysioNet gait dataset rate).
    order : int, optional
        Order of the Butterworth polynomial.  Higher order gives a steeper
        roll-off but also increases ringing near the cutoff.  ``4`` is a
        standard clinical biomechanics choice.  Default ``4``.

    Returns
    -------
    np.ndarray, shape (N,)
        Filtered signal, same length and dtype as *signal*.

    Raises
    ------
    ValueError
        If ``cutoff`` is not strictly between 0 and the Nyquist frequency
        (``fs / 2``).
    """
    nyquist = fs / 2.0
    if not (0 < cutoff < nyquist):
        raise ValueError(
            f"cutoff ({cutoff} Hz) must be strictly between 0 and the "
            f"Nyquist frequency ({nyquist} Hz).  Got fs={fs}."
        )

    # butter() returns normalised digital coefficients when btype='low'
    b, a = butter(order, cutoff / nyquist, btype="low", analog=False)

    # filtfilt pads the signal at both ends to minimise edge transients
    return filtfilt(b, a, signal)


# ---------------------------------------------------------------------------
# 2. Gait-cycle segmentation via peak detection
# ---------------------------------------------------------------------------

def segment_gait_cycles(
    signal: np.ndarray,
    fs: float = 100.0,
    min_step_interval: float = 0.4,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Detect step peaks and extract individual gait-cycle segments.

    Each step produces one clear force peak as the foot bears the body's
    weight (roughly body-weight magnitude).  ``scipy.signal.find_peaks``
    locates these peaks; the signal between two consecutive peaks is then
    returned as one *gait cycle* segment.

    Why peak-to-peak segmentation?
    -------------------------------
    A gait cycle is conventionally defined from one heel-strike to the
    next ipsilateral heel-strike.  For ground-reaction-force data the heel-
    strike coincides closely with the loading peak, making peak-to-peak
    slicing a robust and parameter-light approximation.

    Parameters
    ----------
    signal : np.ndarray, shape (N,)
        Filtered (or raw) 1-D force signal.  It is recommended to pass the
        output of ``lowpass_filter`` so that spurious noise spikes are not
        picked up as false peaks.
    fs : float, optional
        Sampling rate in **Hz**, used to convert ``min_step_interval`` from
        seconds to samples.  Default ``100.0``.
    min_step_interval : float, optional
        Minimum time (seconds) that must separate two consecutive detected
        peaks.  This guards against double-peaks within a single step.
        At normal walking cadence (approx. 100-120 steps/min) the inter-step
        interval is approx. 0.5-0.6 s, so ``0.4`` s gives a safe lower bound
        while still accommodating fast walkers.  Default ``0.4``.

    Returns
    -------
    peaks : np.ndarray, shape (P,)
        Sample indices of the detected step peaks (integer).
    cycles : list of np.ndarray
        Length-(P-1) list; ``cycles[i]`` is the signal slice between
        ``peaks[i]`` and ``peaks[i+1]``, representing the *i*-th gait
        cycle.  Each array may have a different length because stride
        duration varies across steps.

    Notes
    -----
    * The function does **not** further normalise or resample the segments.
      Down-stream feature extraction should handle variable-length cycles
      (e.g., time-normalise to 100 points).
    * If fewer than 2 peaks are detected, ``cycles`` is an empty list.
    """
    # Convert the temporal threshold to samples
    min_distance_samples = int(min_step_interval * fs)

    # Height threshold: peaks must exceed 5 % of the signal's range to
    # filter out baseline noise fluctuations during swing phase.
    signal_range = float(np.ptp(signal))          # peak-to-peak amplitude
    height_threshold = float(np.min(signal)) + 0.05 * signal_range

    peaks, _ = find_peaks(
        signal,
        height=height_threshold,
        distance=min_distance_samples,
    )

    # Slice the signal between consecutive peaks -> one segment per cycle
    cycles: list[np.ndarray] = [
        signal[peaks[i] : peaks[i + 1]]
        for i in range(len(peaks) - 1)
    ]

    return peaks, cycles
