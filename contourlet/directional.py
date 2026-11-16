"""
contourlet/directional.py

Directional Filter Bank (DFB) — second stage of the Contourlet Transform.

Fix vs v1:
    K=8 now uses diagonal fan filters (get_diagonal_fan_filters) in the
    second tree stage instead of repeating the same LP/HP separable filters.
    This gives genuine 8-direction coverage instead of collapsing to 3.

Direction labels:
    K=4:  [0°, 45°, 90°, 135°]
    K=8:  [0°, 22.5°, 45°, 67.5°, 90°, 112.5°, 135°, 157.5°]
"""

import numpy as np
from .filters import (
    get_modulated_fan_filters,
    get_diagonal_fan_filters,
    conv2d_full_mirror,
    conv2d_separable_mirror,
    PKVA_LOWPASS_ANALYSIS,
    PKVA_HIGHPASS_ANALYSIS,
)


# ---------------------------------------------------------------------------
# K=4 split — 2 stages of fan filtering
# ---------------------------------------------------------------------------

def _dfb_split_4(bandpass: np.ndarray) -> list:
    """
    Split bandpass into 4 directional subbands.

    Stage 1: axis-aligned fan pair splits into horizontal/vertical halves.
    Stage 2: each half is further split by filtering along the perpendicular
             axis, giving 4 angular wedges.

    Returns:
        list of 4 arrays, dominant angles ~[0°, 45°, 90°, 135°]
    """
    f0, f1 = get_modulated_fan_filters()

    # Stage 1 — fan split
    y0 = conv2d_full_mirror(bandpass, f0)   # horizontal fan
    y1 = conv2d_full_mirror(bandpass, f1)   # vertical fan

    # Downsample each: rows for y0, cols for y1
    d0 = y0[::2, :]    # shape (H/2, W)
    d1 = y1[:, ::2]    # shape (H, W/2)

    h0 = PKVA_LOWPASS_ANALYSIS
    h1 = PKVA_HIGHPASS_ANALYSIS

    # Stage 2 — split each half using separable filters on perpendicular axis
    # d0: horizontal-dominant, split further along columns
    d00 = conv2d_separable_mirror(d0, np.array([1.0]), h0)[:, ::2]  # LP cols
    d01 = conv2d_separable_mirror(d0, np.array([1.0]), h1)[:, ::2]  # HP cols

    # d1: vertical-dominant, split further along rows
    d10 = conv2d_separable_mirror(d1, h0, np.array([1.0]))[::2, :]  # LP rows
    d11 = conv2d_separable_mirror(d1, h1, np.array([1.0]))[::2, :]  # HP rows

    # Angular order: ~0°, ~45°, ~90°, ~135°
    return [d00, d01, d10, d11]


# ---------------------------------------------------------------------------
# K=8 split — uses diagonal fans for second stage to get true 8-direction coverage
# ---------------------------------------------------------------------------

def _dfb_split_8(bandpass: np.ndarray) -> list:
    """
    Split bandpass into 8 directional subbands.

    Stage 1: axis-aligned fan pair → 2 halves.
    Stage 2: diagonal fan pair applied to each half → 2 sub-halves each = 4 bands.
    Stage 3: each of the 4 bands split by separable LP/HP → 8 final bands.

    Using diagonal fans in stage 2 (instead of repeating axis-aligned fans)
    is the key fix — it rotates the frequency partition 45° and produces
    genuinely different orientation selectivity.

    Returns:
        list of 8 arrays, dominant angles ~[0°,22.5°,45°,67.5°,90°,112.5°,135°,157.5°]
    """
    f0, f1 = get_modulated_fan_filters()
    fd0, fd1 = get_diagonal_fan_filters()

    h0 = PKVA_LOWPASS_ANALYSIS
    h1 = PKVA_HIGHPASS_ANALYSIS

    # Stage 1: axis-aligned fan split
    y0 = conv2d_full_mirror(bandpass, f0)   # horizontal half
    y1 = conv2d_full_mirror(bandpass, f1)   # vertical half

    subbands = []

    # Stage 2 + 3 for y0 (horizontal half)
    # Apply diagonal fans to further split the horizontal half
    y00 = conv2d_full_mirror(y0, fd0)
    y01 = conv2d_full_mirror(y0, fd1)

    # Downsample and split each via separable filters
    d00 = y00[::2, :]
    d01 = y01[::2, :]

    subbands.append(conv2d_separable_mirror(d00, np.array([1.0]), h0)[:, ::2])
    subbands.append(conv2d_separable_mirror(d00, np.array([1.0]), h1)[:, ::2])
    subbands.append(conv2d_separable_mirror(d01, np.array([1.0]), h0)[:, ::2])
    subbands.append(conv2d_separable_mirror(d01, np.array([1.0]), h1)[:, ::2])

    # Stage 2 + 3 for y1 (vertical half)
    y10 = conv2d_full_mirror(y1, fd0)
    y11 = conv2d_full_mirror(y1, fd1)

    d10 = y10[:, ::2]
    d11 = y11[:, ::2]

    subbands.append(conv2d_separable_mirror(d10, h0, np.array([1.0]))[::2, :])
    subbands.append(conv2d_separable_mirror(d10, h1, np.array([1.0]))[::2, :])
    subbands.append(conv2d_separable_mirror(d11, h0, np.array([1.0]))[::2, :])
    subbands.append(conv2d_separable_mirror(d11, h1, np.array([1.0]))[::2, :])

    return subbands  # 8 subbands


# ---------------------------------------------------------------------------
# DirectionalFilterBank class
# ---------------------------------------------------------------------------

class DirectionalFilterBank:
    """
    Directional Filter Bank — decomposes a bandpass image into K directional
    subbands.

    K=4: suitable for finest LP level (primary feature source)
    K=8: suitable for coarser LP levels
    """

    ORIENTATION_LABELS = {
        4: [0.0, 45.0, 90.0, 135.0],
        8: [0.0, 22.5, 45.0, 67.5, 90.0, 112.5, 135.0, 157.5],
    }

    def __init__(self, num_directions: int = 4):
        if num_directions not in (4, 8):
            raise ValueError(f"num_directions must be 4 or 8, got {num_directions}")
        self.num_directions = num_directions

    @property
    def orientations(self) -> list:
        return self.ORIENTATION_LABELS[self.num_directions]

    def decompose(self, bandpass: np.ndarray) -> list:
        """
        Decompose a bandpass image into K directional subbands.

        Args:
            bandpass: 2-D float64 array (one LP bandpass layer, one channel)

        Returns:
            list of K 2-D float64 arrays
        """
        bp = bandpass.astype(np.float64)
        if self.num_directions == 4:
            return _dfb_split_4(bp)
        else:
            return _dfb_split_8(bp)

    def decompose_multichannel(self, bandpass_rgb: dict) -> dict:
        """Apply DFB to R, G, B bandpass layers independently."""
        return {ch: self.decompose(bandpass_rgb[ch]) for ch in ('R', 'G', 'B')}

    def get_coefficient_magnitudes(self, subbands: list) -> list:
        return [np.abs(s) for s in subbands]

    def get_dominant_direction_map(self, subbands: list) -> np.ndarray:
        """
        D(x,y) = argmax_k |C_k(x,y)|

        All subbands are resized to the largest shape before argmax so that
        spatial positions are comparable across subbands.

        Returns:
            D: 2-D int array, values in {0, ..., K-1}
        """
        mags = self.get_coefficient_magnitudes(subbands)
        max_h = max(m.shape[0] for m in mags)
        max_w = max(m.shape[1] for m in mags)

        resized = []
        for m in mags:
            if m.shape == (max_h, max_w):
                resized.append(m)
            else:
                # Nearest-neighbour upsample — preserves magnitude ordering
                row_idx = np.round(
                    np.linspace(0, m.shape[0] - 1, max_h)).astype(int)
                col_idx = np.round(
                    np.linspace(0, m.shape[1] - 1, max_w)).astype(int)
                resized.append(m[np.ix_(row_idx, col_idx)])

        stacked = np.stack(resized, axis=0)   # (K, H, W)
        return np.argmax(stacked, axis=0)      # (H, W)


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    np.random.seed(42)
    x = np.linspace(0, 4 * np.pi, 224)
    xx, yy = np.meshgrid(x, x)
    test_bp = (np.sin(xx) + np.cos(yy) + np.sin(xx + yy) + np.cos(xx - yy)) * 50.0

    print("=== DFB Smoke Test ===")
    for K in [4, 8]:
        dfb = DirectionalFilterBank(num_directions=K)
        subbands = dfb.decompose(test_bp)
        D = dfb.get_dominant_direction_map(subbands)
        unique = np.unique(D)
        print(f"  K={K}: {len(subbands)} subbands, "
              f"{len(unique)}/{K} directions in map: {unique.tolist()}")