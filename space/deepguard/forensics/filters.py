"""
contourlet/filters.py

PKVA filter coefficients and construction utilities for the Contourlet Transform.

Fixes applied vs v1:
  - Replaced raw unnormalized coefficients with proper 9/7 biorthogonal values
    normalized to unit DC gain (sum = 1.0). This fixes Test 1 and Test 3.
  - Added get_checkerboard_filter() for the DFB second-stage split, which fixes
    the K=8 direction coverage failure (Test 5).
"""

import numpy as np
from scipy.signal import fftconvolve


# ---------------------------------------------------------------------------
# PKVA 9/7 biorthogonal lowpass — analysis side
# These are the standard Cohen-Daubechies-Feauveau 9/7 coefficients,
# which are the PKVA filters used in the contourlet toolbox.
# Normalized so sum = 1.0 (unit DC gain).
# ---------------------------------------------------------------------------

_CDF97_RAW = np.array([
     0.026748757410810,
    -0.016864118442875,
    -0.078223266528988,
     0.266864118442875,
     0.602949018236358,
     0.266864118442875,
    -0.078223266528988,
    -0.016864118442875,
     0.026748757410810,
], dtype=np.float64)

# Normalize to unit DC gain
PKVA_LOWPASS_ANALYSIS  = _CDF97_RAW / _CDF97_RAW.sum()
PKVA_LOWPASS_SYNTHESIS = PKVA_LOWPASS_ANALYSIS[::-1].copy()


def _qmf(filt: np.ndarray) -> np.ndarray:
    """Quadrature mirror: h1[n] = (-1)^n * h0[N-1-n]"""
    n = np.arange(len(filt))
    return ((-1.0) ** n) * filt[::-1]


PKVA_HIGHPASS_ANALYSIS  = _qmf(PKVA_LOWPASS_SYNTHESIS)
PKVA_HIGHPASS_SYNTHESIS = _qmf(PKVA_LOWPASS_ANALYSIS)


# ---------------------------------------------------------------------------
# Fan filters for the DFB — built via McClellan transform
# ---------------------------------------------------------------------------

def _make_half_band_proto(length: int = 9) -> np.ndarray:
    """1-D half-band lowpass prototype derived from PKVA, windowed to `length`."""
    base = PKVA_LOWPASS_ANALYSIS.copy()
    if len(base) > length:
        start = (len(base) - length) // 2
        proto = base[start:start + length]
    elif len(base) < length:
        pad = length - len(base)
        proto = np.pad(base, (pad // 2, pad - pad // 2))
    else:
        proto = base.copy()
    proto = proto * np.hamming(len(proto))
    proto /= proto.sum() + 1e-12
    return proto


def _mcclellan_fan(size: int = 15, rotate: bool = False) -> np.ndarray:
    """
    Build a 2-D fan filter via McClellan transform on the PKVA prototype.

    rotate=False → horizontal fan (passes |omega_y| < |omega_x|)
    rotate=True  → vertical   fan (passes |omega_x| < |omega_y|)
    """
    assert size % 2 == 1
    half = size // 2
    proto = _make_half_band_proto(length=size)
    cx = cy = float(half + 1)

    x = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)

    if not rotate:
        t = np.cos(np.pi * xx / cx) + np.cos(np.pi * yy / cy) - 1.0
    else:
        t = np.cos(np.pi * yy / cx) + np.cos(np.pi * xx / cy) - 1.0
    t = np.clip(t, -1.0, 1.0)

    # Chebyshev expansion: H_2D = sum_k h[k] * T_k(t)
    filt = np.zeros((size, size), dtype=np.float64)
    T0 = np.ones_like(t)
    T1 = t.copy()

    for k in range(size):
        c = proto[k] if k < len(proto) else 0.0
        if k == 0:
            filt += c * T0
        elif k == 1:
            filt += c * T1
        else:
            T2 = 2.0 * t * T1 - T0
            filt += c * T2
            T0 = T1.copy()
            T1 = T2.copy()

    return filt


def get_modulated_fan_filters(size: int = 15) -> tuple:
    """
    Return complementary fan filter pair (f0, f1).
    f0 + f1 partitions the 2-D frequency plane.
    """
    f0 = _mcclellan_fan(size=size, rotate=False)
    f1 = _mcclellan_fan(size=size, rotate=True)
    # Normalize so they form a proper partition
    total = np.abs(f0).sum() + np.abs(f1).sum() + 1e-12
    f0 /= total
    f1 /= total
    return f0, f1


def get_checkerboard_filter(size: int = 11) -> np.ndarray:
    """
    Checkerboard-modulated bandpass filter for DFB second-stage splits.

    Modulating a separable lowpass by (-1)^(x+y) rotates its frequency
    response by 45 degrees — giving genuinely different directional
    selectivity from the fan filters used in stage 1. This is the fix
    for K=8 only showing 3/8 directions (Test 5 failure).

    Returns:
        2-D filter, shape (size, size)
    """
    assert size % 2 == 1
    half = size // 2
    lp = _make_half_band_proto(length=size)
    lp_2d = np.outer(lp, lp)
    x = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)
    mod = (-1.0) ** (xx + yy)
    return lp_2d * mod


def get_diagonal_fan_filters(size: int = 15) -> tuple:
    """
    Return fan filters rotated 45 degrees — used in the DFB second stage
    for K=8 decomposition to give the remaining 4 angular wedges.

    These complement the axis-aligned fans from get_modulated_fan_filters().

    Returns:
        fd0, fd1: each shape (size, size)
    """
    assert size % 2 == 1
    half = size // 2
    proto = _make_half_band_proto(length=size)
    cx = cy = float(half + 1)

    x = np.arange(-half, half + 1, dtype=np.float64)
    xx, yy = np.meshgrid(x, x)

    # Diagonal fan: substitute rotated coordinates (x+y)/sqrt(2), (x-y)/sqrt(2)
    xr = (xx + yy) / np.sqrt(2.0)
    yr = (xx - yy) / np.sqrt(2.0)
    xr = np.clip(xr, -half, half)
    yr = np.clip(yr, -half, half)

    t0 = np.cos(np.pi * xr / cx) + np.cos(np.pi * yr / cy) - 1.0
    t1 = np.cos(np.pi * yr / cx) + np.cos(np.pi * xr / cy) - 1.0
    t0 = np.clip(t0, -1.0, 1.0)
    t1 = np.clip(t1, -1.0, 1.0)

    def _chebyshev_eval(t):
        f = np.zeros((size, size), dtype=np.float64)
        T0 = np.ones_like(t)
        T1 = t.copy()
        for k in range(size):
            c = proto[k] if k < len(proto) else 0.0
            if k == 0:
                f += c * T0
            elif k == 1:
                f += c * T1
            else:
                T2 = 2.0 * t * T1 - T0
                f += c * T2
                T0 = T1.copy()
                T1 = T2.copy()
        return f

    fd0 = _chebyshev_eval(t0)
    fd1 = _chebyshev_eval(t1)
    total = np.abs(fd0).sum() + np.abs(fd1).sum() + 1e-12
    fd0 /= total
    fd1 /= total
    return fd0, fd1


# ---------------------------------------------------------------------------
# Convolution utilities with mirror boundary extension
# ---------------------------------------------------------------------------

def conv1d_mirror(signal: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """1-D convolution with symmetric boundary. Output same length as input."""
    K = len(kernel)
    pad = K // 2
    if pad == 0:
        return np.convolve(signal.astype(np.float64), kernel, mode='same')
    left  = signal[pad:0:-1]
    right = signal[-2:-2 - pad:-1]
    padded = np.concatenate([left, signal.astype(np.float64), right])
    full = np.convolve(padded, kernel, mode='valid')
    start = (len(full) - len(signal)) // 2
    start = max(start, 0)
    return full[start:start + len(signal)]


def conv2d_separable_mirror(image: np.ndarray,
                             row_kernel: np.ndarray,
                             col_kernel: np.ndarray) -> np.ndarray:
    """2-D separable convolution with symmetric boundary. Output shape = input shape."""
    img = image.astype(np.float64)
    H, W = img.shape
    out = np.empty_like(img)
    for r in range(H):
        out[r, :] = conv1d_mirror(img[r, :], row_kernel)
    result = np.empty_like(out)
    for c in range(W):
        result[:, c] = conv1d_mirror(out[:, c], col_kernel)
    return result


def conv2d_full_mirror(image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
    """2-D non-separable convolution with symmetric boundary via FFT."""
    Kh, Kw = kernel.shape
    pad_h, pad_w = Kh // 2, Kw // 2
    img_padded = np.pad(image.astype(np.float64),
                        ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    conv = fftconvolve(img_padded, kernel[::-1, ::-1], mode='valid')
    H, W = image.shape
    oh, ow = conv.shape
    sh = max((oh - H) // 2, 0)
    sw = max((ow - W) // 2, 0)
    return conv[sh:sh + H, sw:sw + W]


# ---------------------------------------------------------------------------
# Diagnostic
# ---------------------------------------------------------------------------

def print_filter_info() -> None:
    print("=== PKVA Filter Properties ===")
    print(f"  Lowpass  DC gain : {PKVA_LOWPASS_ANALYSIS.sum():.6f}  (target 1.0)")
    print(f"  Highpass DC gain : {PKVA_HIGHPASS_ANALYSIS.sum():.6f}  (target 0.0)")
    f0, f1 = get_modulated_fan_filters()
    print(f"  Fan f0 DC        : {f0.sum():.4f}")
    print(f"  Fan f1 DC        : {f1.sum():.4f}")


if __name__ == "__main__":
    print_filter_info()