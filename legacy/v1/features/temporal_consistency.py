"""
features/temporal_consistency.py

Temporal Consistency Feature — pixel-level std across 20 aligned frames.

Real faces: smooth, physically consistent appearance → LOW std
Fake faces: frame-independent GAN synthesis → HIGHER std in artifact regions

Returns (224, 224, 3) float32 normalized std map.
Compression robust: relative frame-to-frame variation preserved under JPEG.
Novel: no published paper uses this as a dedicated detection branch.
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List


def extract_temporal_consistency(
    aligned_frame_paths: List[Path],
    n_frames: int = 20,
) -> np.ndarray:
    if not aligned_frame_paths:
        return np.zeros((224, 224, 3), dtype=np.float32)

    total = len(aligned_frame_paths)
    if total <= n_frames:
        selected = aligned_frame_paths
    else:
        indices  = np.linspace(0, total-1, n_frames, dtype=int)
        selected = [aligned_frame_paths[i] for i in indices]

    frames = []
    for path in selected:
        try:
            img = np.array(Image.open(path).convert("RGB")).astype(np.float32)
            if img.shape == (224, 224, 3):
                frames.append(img)
        except Exception:
            continue

    if len(frames) < 2:
        return np.zeros((224, 224, 3), dtype=np.float32)

    stack   = np.stack(frames, axis=0)   # (N, 224, 224, 3)
    std_map = stack.std(axis=0)           # (224, 224, 3)

    max_val = std_map.max()
    if max_val > 1e-6:
        std_map = std_map / max_val

    return std_map.astype(np.float32)


def feature_shape():
    return (224, 224, 3)