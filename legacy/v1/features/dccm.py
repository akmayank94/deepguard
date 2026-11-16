"""
features/dccm.py — final

Fix: removed per-image standardization from DCCM differences.
Raw COM differences are preserved — the precompute_features.py will
fit a StandardScaler across the full dataset and save it alongside
the cache, so normalization happens at dataset level not image level.
"""

import numpy as np
from contourlet.filters import (
    PKVA_LOWPASS_ANALYSIS, PKVA_HIGHPASS_ANALYSIS, conv2d_separable_mirror)

SPATIAL_OFFSETS = [(0, 1), (1, 0), (1, 1), (1, -1)]
COM_SIZE        = 3   # 3 HF directions
EPS             = 1e-8


def _hf_maps(channel: np.ndarray) -> list:
    """Returns [LH, HL, HH] at full resolution."""
    h0, h1 = PKVA_LOWPASS_ANALYSIS, PKVA_HIGHPASS_ANALYSIS
    ch = channel.astype(np.float64)
    return [conv2d_separable_mirror(ch, h0, h1),
            conv2d_separable_mirror(ch, h1, h0),
            conv2d_separable_mirror(ch, h1, h1)]


def _direction_map(channel: np.ndarray) -> np.ndarray:
    """D(x,y) = argmax_k |HF_k(x,y)|. Values in {0,1,2}."""
    maps = _hf_maps(channel)
    return np.argmax(np.stack([np.abs(m) for m in maps], axis=0), axis=0)


def _com(D: np.ndarray, offset: tuple, K: int = COM_SIZE) -> np.ndarray:
    dr, dc = offset
    H, W = D.shape
    com = np.zeros((K, K), dtype=np.float64)
    r1s = max(0, dr);  r1e = H - max(0, -dr)
    c1s = max(0, dc);  c1e = W - max(0, -dc)
    r2s = max(0, -dr); r2e = H - max(0, dr)
    c2s = max(0, -dc); c2e = W - max(0, dc)
    np.add.at(com,
              (D[r1s:r1e, c1s:c1e].ravel(), D[r2s:r2e, c2s:c2e].ravel()), 1)
    com += com.T
    total = com.sum()
    if total > EPS:
        com /= total
    return com


class DCCMExtractor:
    def __init__(self, offsets=None, ct=None):
        self.offsets = offsets or SPATIAL_OFFSETS

    def extract(self, image: np.ndarray) -> dict:
        img = image.astype(np.float64)
        if img.ndim == 2:
            img = np.stack([img] * 3, axis=-1)
        if img.max() <= 1.0 + 1e-6:
            img *= 255.0

        D = {ch: _direction_map(img[:, :, i])
             for i, ch in enumerate(('R', 'G', 'B'))}

        parts = []
        dccm_diffs = {}

        for oi, offset in enumerate(self.offsets):
            coms = {ch: _com(D[ch], offset) for ch in ('R', 'G', 'B')}
            diffs = {
                'RB': coms['R'] - coms['B'],
                'RG': coms['R'] - coms['G'],
                'GB': coms['G'] - coms['B'],
            }
            dccm_diffs[oi] = diffs
            for name in ('RB', 'RG', 'GB'):
                # Raw differences — no per-image normalization
                parts.append(diffs[name].flatten().astype(np.float32))

        return {
            'feature_vector':  np.concatenate(parts),
            'dccm_diffs':      dccm_diffs,
            'direction_maps':  D,
        }

    def feature_dim(self) -> int:
        return len(self.offsets) * 3 * (COM_SIZE ** 2)