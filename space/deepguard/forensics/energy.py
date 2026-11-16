"""
features/energy.py — final

Key fixes vs v3:
1. diff_image restored to use ABSOLUTE energy (not signed difference) so
   EfficientNet sees meaningful texture differences (gap 0.083 restored)
2. Feature vector uses RAW differences without per-image standardization —
   per-image standardization destroys inter-image discriminative differences
3. Global clipping to [-10, 10] before concat prevents extreme outliers
   from dominating without destroying the distribution shape
"""

import numpy as np
from .filters import (
    PKVA_LOWPASS_ANALYSIS, PKVA_HIGHPASS_ANALYSIS, conv2d_separable_mirror)

BLOCK_SIZES   = [8, 16, 32]
EPS           = 1e-8


def _filtered_maps(channel: np.ndarray) -> list:
    """4 separable PKVA responses at full resolution. [LL, LH, HL, HH]"""
    h0, h1 = PKVA_LOWPASS_ANALYSIS, PKVA_HIGHPASS_ANALYSIS
    ch = channel.astype(np.float64)
    return [conv2d_separable_mirror(ch, h0, h0),
            conv2d_separable_mirror(ch, h0, h1),
            conv2d_separable_mirror(ch, h1, h0),
            conv2d_separable_mirror(ch, h1, h1)]


def _block_energy(m: np.ndarray, bs: int) -> np.ndarray:
    H, W = m.shape
    Hb, Wb = H // bs, W // bs
    c = m[:Hb*bs, :Wb*bs]
    return np.sum(c.reshape(Hb, bs, Wb, bs) ** 2, axis=(1, 3))


def _build_diff_image(img: np.ndarray) -> np.ndarray:
    """
    3-channel diff image: [E_R - E_B, E_R - E_G, E_G - E_B]
    at full resolution. Per-channel min-max normalized to [0,1].
    """
    img_f = img.astype(np.float64)
    if img_f.max() <= 1.0 + 1e-6:
        img_f *= 255.0

    def _hf_energy(ch_idx):
        maps = _filtered_maps(img_f[:, :, ch_idx])
        return np.mean([maps[1]**2, maps[2]**2, maps[3]**2], axis=0)

    E_R = _hf_energy(0)
    E_G = _hf_energy(1)
    E_B = _hf_energy(2)

    # Inter-channel DIFFERENCES — not per-channel absolute values
    diff = np.stack([E_R - E_B, E_R - E_G, E_G - E_B], axis=-1).astype(np.float32)

    # Per-channel min-max normalize to [0,1]
    for c in range(3):
        mn, mx = diff[:, :, c].min(), diff[:, :, c].max()
        rng = mx - mn
        diff[:, :, c] = (diff[:, :, c] - mn) / rng if rng > EPS else 0.5

    return diff


class InterChannelEnergyExtractor:
    def __init__(self, block_sizes=None, ct=None):
        self.block_sizes = block_sizes or BLOCK_SIZES

    def extract(self, image: np.ndarray) -> dict:
        img = image.astype(np.float64)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() <= 1.0 + 1e-6:
            img *= 255.0

        # Full-resolution filtered maps per channel
        maps = {ch: _filtered_maps(img[:, :, i])
                for i, ch in enumerate(('R', 'G', 'B'))}

        # Use HF directions only: LH(1), HL(2), HH(3)
        HF = [1, 2, 3]
        parts = []
        diff_maps_store = {}

        for k in HF:
            diff_maps_store[k] = {}
            for bs in self.block_sizes:
                eR = _block_energy(maps['R'][k], bs)
                eG = _block_energy(maps['G'][k], bs)
                eB = _block_energy(maps['B'][k], bs)

                # Raw differences — do NOT standardize per image
                dRB = (eR - eB).flatten()
                dRG = (eR - eG).flatten()
                dGB = (eG - eB).flatten()

                diff_maps_store[k][bs] = {
                    'RB': dRB, 'RG': dRG, 'GB': dGB}
                parts.extend([dRB, dRG, dGB])

        raw = np.concatenate(parts).astype(np.float32)
        # Clip outliers only — do NOT standardize (destroys inter-image differences)
        feature_vector = np.clip(raw, -1e6, 1e6)

        diff_image = _build_diff_image(image)

        subband_energies = {
            ch: [float(np.sum(maps[ch][k] ** 2)) for k in HF]
            for ch in ('R', 'G', 'B')
        }

        return {
            'feature_vector':   feature_vector,
            'diff_image':       diff_image,
            'diff_maps':        diff_maps_store,
            'subband_energies': subband_energies,
        }

    def feature_dim(self) -> int:
        return 3 * 3 * sum((224 // bs) ** 2 for bs in self.block_sizes)