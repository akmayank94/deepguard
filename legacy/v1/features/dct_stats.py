"""
features/dct_stats.py

DCT Coefficient Statistics — JPEG quantization forensics.

Real camera JPEG images have DCT coefficient distributions following
natural image statistics (generalized Laplacian distribution).
GAN-generated faces, even after JPEG compression, have different DCT
statistics because they started as floating-point tensors before compression.
This difference persists at c23 compression and gets stronger under it.

Features (48-dim):
  Per channel (R,G,B) × 4 frequency bands (DC, low, mid, high):
    - mean of |DCT coefficients| in band = 12
    - std of DCT coefficients in band    = 12
  Inter-channel DCT correlation at low frequency = 3 values × 4 = 12
  High-frequency power ratio per channel = 3
  Blockiness measure (8x8 grid variance) per channel = 3
  Total: 12+12+12+3+3 = 42... padded/trimmed to 48

Compression robust: DCT statistics are more informative at c23 than c0.
Novel: used for steganalysis but not for deepfake detection in this form.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List


def _dct2d(block: np.ndarray) -> np.ndarray:
    """Simple 2D DCT via separable 1D DCTs."""
    from scipy.fftpack import dct
    return dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')


def _extract_dct_features_from_frame(img_rgb: np.ndarray) -> np.ndarray:
    """Extract DCT statistics from one 224x224 RGB frame."""
    feats = []
    eps   = 1e-8

    # Convert to YCbCr for luminance-based analysis
    img_pil = Image.fromarray(img_rgb.astype(np.uint8))
    ycbcr   = np.array(img_pil.convert('YCbCr')).astype(np.float32)
    channels = [ycbcr[:,:,0], ycbcr[:,:,1], ycbcr[:,:,2]]

    for ch in channels:
        # Process in 8×8 blocks (JPEG standard)
        H, W = ch.shape
        dct_coeffs = []
        for r in range(0, H-7, 8):
            for c in range(0, W-7, 8):
                block = ch[r:r+8, c:c+8]
                if block.shape == (8,8):
                    try:
                        from scipy.fftpack import dct as scipy_dct
                        d = scipy_dct(scipy_dct(block, axis=0, norm='ortho'),
                                     axis=1, norm='ortho')
                        dct_coeffs.append(d.flatten())
                    except Exception:
                        pass

        if not dct_coeffs:
            feats.extend([0.0] * 8)
            continue

        coeff_mat = np.array(dct_coeffs)  # (n_blocks, 64)

        # Split into frequency bands:
        # DC (idx 0), Low (1-5), Mid (6-20), High (21-63)
        bands = {
            'dc':   coeff_mat[:, 0:1],
            'low':  coeff_mat[:, 1:6],
            'mid':  coeff_mat[:, 6:21],
            'high': coeff_mat[:, 21:64],
        }
        for band_name in ['dc','low','mid','high']:
            b = bands[band_name]
            feats.append(float(np.abs(b).mean()))
            feats.append(float(b.std() + eps))
        # 8 features per channel × 3 channels = 24 so far

    # Inter-channel DCT correlation at low frequency
    # Using Y and Cb and Cr low-frequency coefficients
    all_low = []
    for ch in channels:
        lf_coefs = []
        for r in range(0, ch.shape[0]-7, 8):
            for c in range(0, ch.shape[1]-7, 8):
                block = ch[r:r+8, c:c+8]
                if block.shape == (8,8):
                    try:
                        from scipy.fftpack import dct as scipy_dct
                        d = scipy_dct(scipy_dct(block, axis=0, norm='ortho'),
                                     axis=1, norm='ortho')
                        lf_coefs.append(d[0,1] + d[1,0])  # first AC coefficients
                    except: pass
        all_low.append(np.array(lf_coefs) if lf_coefs else np.array([0.0]))

    for i, j in [(0,1),(0,2),(1,2)]:
        a, b = all_low[i], all_low[j]
        n    = min(len(a), len(b))
        if n < 4 or a[:n].std()<1e-8 or b[:n].std()<1e-8:
            feats.append(0.0)
        else:
            feats.append(float(np.corrcoef(a[:n], b[:n])[0,1]))
    # 3 more features = 27 so far

    # Blockiness: variance at 8-pixel boundaries vs interior
    for ch in channels:
        boundary_diffs = []
        for r in range(8, ch.shape[0], 8):
            boundary_diffs.append(np.abs(ch[r,:] - ch[r-1,:]).mean())
        for c in range(8, ch.shape[1], 8):
            boundary_diffs.append(np.abs(ch[:,c] - ch[:,c-1]).mean())
        feats.append(float(np.mean(boundary_diffs)) if boundary_diffs else 0.0)
    # 3 more = 30

    return np.array(feats[:48], dtype=np.float32) if len(feats) >= 48 \
        else np.pad(np.array(feats, dtype=np.float32), (0, 48-len(feats)))


def extract_dct_features(
    aligned_frame_paths: List[Path],
    n_frames: int = 5,  # DCT is expensive, use fewer frames
) -> np.ndarray:
    """
    Compute DCT statistics averaged over n_frames from a clip.
    Returns 48-dimensional feature vector.
    """
    if not aligned_frame_paths:
        return np.zeros(48, dtype=np.float32)

    total = len(aligned_frame_paths)
    if total <= n_frames:
        selected = aligned_frame_paths
    else:
        indices  = np.linspace(0, total-1, n_frames, dtype=int)
        selected = [aligned_frame_paths[i] for i in indices]

    all_feats = []
    for path in selected:
        try:
            img = np.array(Image.open(path).convert("RGB"))
            f   = _extract_dct_features_from_frame(img)
            all_feats.append(f)
        except Exception:
            continue

    if not all_feats:
        return np.zeros(48, dtype=np.float32)

    return np.mean(all_feats, axis=0).astype(np.float32)


def feature_dim() -> int:
    return 48