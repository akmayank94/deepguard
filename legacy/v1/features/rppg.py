"""
features/rppg.py

rPPG (Remote PhotoPlethysmoGraphy) feature extraction.

Real human faces have blood flowing through skin. As the heart beats,
blood volume in skin capillaries changes periodically. This changes
how much green light (most sensitive to blood) the skin absorbs/reflects.
The result: a tiny periodic variation in the green channel intensity
at heartbeat frequency (0.75 - 2.5 Hz = 45-150 BPM).

GAN-generated faces:
  - Have NO real blood flow → no heartbeat frequency
  - May have temporal noise or GAN-specific frequency artifacts
  - The power spectrum at heartbeat frequencies is different from real faces

Our preprocessing captures:
  - 64 frames at effective 12.5 Hz sampling rate
  - 5.12 seconds of signal
  - Frequency resolution: 12.5/64 = 0.195 Hz per FFT bin

Heartbeat frequency range: 0.75 - 2.5 Hz
  - At 12.5 Hz sampling, this covers bins 4-13 out of 32 unique bins
  - We have good resolution in the heartbeat band

Feature extraction:
  1. Load rppg.npy — shape (64, 3) — mean R,G,B per frame
  2. Detrend the signal (remove slow lighting changes)
  3. Apply Hanning window (reduce spectral leakage)
  4. Compute FFT → power spectrum
  5. Extract frequency-domain features:
     - Power in heartbeat band (0.75-2.5 Hz) per channel
     - Peak frequency in heartbeat band per channel
     - Ratio of heartbeat band power to total power
     - Inter-channel spectral correlation in heartbeat band

Output: 32-dimensional feature vector
"""

import numpy as np
from pathlib import Path
from typing import Optional


# Sampling parameters (must match preprocessing)
EFFECTIVE_FPS_25 = 12.5   # for 25fps source (RAW_GAP=2)
EFFECTIVE_FPS_30 = 15.0   # for 30fps source (RAW_GAP=2)
N_FRAMES         = 64

# Heartbeat frequency band
HB_LOW_HZ  = 0.75
HB_HIGH_HZ = 2.5

# Respiratory frequency band (secondary signal)
RESP_LOW_HZ  = 0.15
RESP_HIGH_HZ = 0.5


def _get_fps(rppg_path: Path) -> float:
    """Try to read fps from meta.json, default to 12.5."""
    meta_path = rppg_path.parent / "meta.json"
    if meta_path.exists():
        import json
        try:
            meta = json.load(open(str(meta_path)))
            return float(meta.get('effective_fps_rppg', EFFECTIVE_FPS_25))
        except Exception:
            pass
    return EFFECTIVE_FPS_25


def _power_in_band(power_spectrum: np.ndarray,
                   freqs: np.ndarray,
                   low: float, high: float) -> float:
    """Sum of power spectrum values within [low, high] Hz."""
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return 0.0
    return float(power_spectrum[mask].sum())


def _peak_freq_in_band(power_spectrum: np.ndarray,
                        freqs: np.ndarray,
                        low: float, high: float) -> float:
    """Frequency of peak power within [low, high] Hz."""
    mask = (freqs >= low) & (freqs <= high)
    if not mask.any():
        return float((low + high) / 2)
    return float(freqs[mask][power_spectrum[mask].argmax()])


def extract_rppg_features(rppg_path: Path) -> np.ndarray:
    """
    Extract rPPG frequency features from rppg.npy file.

    Args:
        rppg_path: Path to rppg.npy file — shape (64, 3) float32

    Returns:
        feature vector of shape (32,) float32
    """
    try:
        rppg = np.load(str(rppg_path)).astype(np.float64)  # (64, 3)
    except Exception:
        return np.zeros(32, dtype=np.float32)

    if rppg.shape[0] < 8:
        return np.zeros(32, dtype=np.float32)

    n    = rppg.shape[0]
    fps  = _get_fps(rppg_path)
    feats = []

    # Per-channel analysis (R=0, G=1, B=2)
    per_channel_peaks = []
    per_channel_hb_powers = []

    for ch in range(3):
        signal = rppg[:, ch].copy()

        # 1. Detrend — remove linear trend (slow illumination changes)
        x      = np.arange(n, dtype=np.float64)
        coeffs = np.polyfit(x, signal, 1)
        signal = signal - np.polyval(coeffs, x)

        # 2. Normalize
        std = signal.std()
        if std < 1e-8:
            signal = np.zeros_like(signal)
        else:
            signal = signal / std

        # 3. Hanning window — reduce spectral leakage
        window = np.hanning(n)
        signal = signal * window

        # 4. FFT
        fft_vals = np.fft.rfft(signal)
        power    = np.abs(fft_vals) ** 2
        freqs    = np.fft.rfftfreq(n, d=1.0/fps)

        total_power = power.sum() + 1e-10

        # 5. Heartbeat band features
        hb_power     = _power_in_band(power, freqs, HB_LOW_HZ, HB_HIGH_HZ)
        hb_ratio     = hb_power / total_power
        hb_peak_freq = _peak_freq_in_band(power, freqs, HB_LOW_HZ, HB_HIGH_HZ)

        # 6. Respiratory band features
        resp_power = _power_in_band(power, freqs, RESP_LOW_HZ, RESP_HIGH_HZ)
        resp_ratio = resp_power / total_power

        # 7. High-frequency noise ratio (above 2.5 Hz — should be low in real faces)
        hf_power = _power_in_band(power, freqs, HB_HIGH_HZ, fps/2)
        hf_ratio = hf_power / total_power

        per_channel_peaks.append(hb_peak_freq)
        per_channel_hb_powers.append(hb_power)

        feats.extend([
            float(hb_power),      # raw heartbeat power
            float(hb_ratio),      # heartbeat / total ratio
            float(hb_peak_freq),  # peak frequency in heartbeat band
            float(resp_power),    # respiratory power
            float(resp_ratio),    # respiratory / total ratio
            float(hf_ratio),      # high-freq noise ratio
            float(signal.std()),  # post-detrend temporal variance
        ])
        # 7 features × 3 channels = 21 features so far

    # Inter-channel spectral consistency features
    # Real faces: heartbeat peak should be at similar frequency in R, G, B
    # Fake faces: peaks may be at different frequencies or absent
    r_peak, g_peak, b_peak = per_channel_peaks
    feats.append(float(abs(r_peak - g_peak)))   # R-G peak diff
    feats.append(float(abs(r_peak - b_peak)))   # R-B peak diff
    feats.append(float(abs(g_peak - b_peak)))   # G-B peak diff

    # Power ratio between channels
    r_pow, g_pow, b_pow = per_channel_hb_powers
    total_hb = r_pow + g_pow + b_pow + 1e-10
    feats.append(float(g_pow / total_hb))       # green dominance in HB band
    feats.append(float((r_pow + b_pow) / (2 * g_pow + 1e-10)))  # RB/G ratio

    # Signal-to-noise ratio proxy: heartbeat band vs noise band
    # (computed on green channel — most sensitive to blood)
    g_signal = rppg[:, 1].copy()
    g_signal = g_signal - np.polyval(np.polyfit(np.arange(n), g_signal, 1),
                                      np.arange(n))
    g_std = g_signal.std()
    feats.append(float(g_std))    # overall green temporal variation

    # Dominant frequency across all channels (majority vote proxy)
    feats.append(float(np.median(per_channel_peaks)))

    # Pad/trim to exactly 32 features
    feats = feats[:32]
    while len(feats) < 32:
        feats.append(0.0)

    return np.array(feats, dtype=np.float32)


def feature_dim() -> int:
    return 32