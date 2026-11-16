"""Two alignment questions from the Phase 1.1 audit.

1. Width (D-017.3, D-019.2): were the 282 narrower Face2Face/NeuralTextures fakes
   made by *cropping* the original (at which column offset) or by *rescaling* it?
   We compare the fake frame with each candidate transform of the original, on
   background pixels only (the face is manipulated, so it is excluded).
2. DFD time offset (D-017.6, D-019.4): DFD fakes are trimmed sub-clips of their
   originals. We find the start frame by matching small grayscale frames.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


def _bg_mask(shape: tuple[int, int], boxes: list[np.ndarray], pad: float = 0.25) -> np.ndarray:
    h, w = shape
    m = np.ones((h, w), bool)
    for b in boxes:
        bw, bh = b[2] - b[0], b[3] - b[1]
        x1, y1 = int(max(0, b[0] - pad * bw)), int(max(0, b[1] - pad * bh))
        x2, y2 = int(min(w, b[2] + pad * bw)), int(min(h, b[3] + pad * bh))
        m[y1:y2, x1:x2] = False
    return m


def width_candidates(orig_rgb: np.ndarray, fake_rgb: np.ndarray,
                     face_boxes_fake: list[np.ndarray]) -> dict:
    """Background MAD of the fake against every crop offset and against rescaling.

    face_boxes_fake are boxes in fake-frame coordinates; they are excluded.
    Returns {"crop": {offset: mad}, "rescale": mad, "best": (mode, offset, mad)}.
    """
    h, w = orig_rgb.shape[:2]
    hf, wf = fake_rgb.shape[:2]
    if hf != h or wf >= w:
        raise ValueError(f"expected same height and narrower fake, got {orig_rgb.shape} "
                         f"vs {fake_rgb.shape}")
    fg = cv2.cvtColor(fake_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    og = cv2.cvtColor(orig_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    mask = _bg_mask((hf, wf), face_boxes_fake)
    crop = {}
    for o in range(0, w - wf + 1):
        crop[o] = float(np.abs(og[:, o:o + wf] - fg)[mask].mean())
    resc = cv2.resize(orig_rgb, (wf, h), interpolation=cv2.INTER_AREA)
    rg = cv2.cvtColor(resc, cv2.COLOR_RGB2GRAY).astype(np.float32)
    rescale = float(np.abs(rg - fg)[mask].mean())
    o_best = min(crop, key=crop.get)
    best = ("crop", o_best, crop[o_best]) if crop[o_best] <= rescale else ("rescale", 0, rescale)
    return {"crop": crop, "rescale": rescale, "best": best}


def crop_offset_rule(width_orig: int, offset: int, multiple: int = 32) -> str:
    """Name the crop offset relative to the removed columns d = w - floor32(w)."""
    d = width_orig - (width_orig // multiple) * multiple
    if offset == 0:
        return "left"
    if offset == d:
        return "right"
    if offset in (d // 2, (d + 1) // 2):
        return "center"
    return "other"


@dataclass
class OffsetResult:
    offset: int          # fake frame j corresponds to original frame j + offset
    err: float           # mean abs gray difference at the best offset
    second_err: float    # best error at offsets more than 2 frames away
    n_probe: int

    @property
    def margin(self) -> float:
        return self.second_err / max(self.err, 1e-6)


def find_offset(fake_small: np.ndarray, orig_small: np.ndarray, n_probe: int = 16,
                min_valid: float = 0.5) -> OffsetResult:
    """Find where the fake starts inside its original.

    fake_small: (Lf, h, w) uint8 small gray frames; orig_small: (Lo, h, w).
    Every offset o in [0, Lo - 1] is scored on n_probe fake frames spread over the
    fake; probes that fall past the end of the original are skipped, and offsets
    with fewer than min_valid x n_probe usable probes are not considered.
    """
    lf, lo = len(fake_small), len(orig_small)
    if lf == 0 or lo == 0:
        raise ValueError("empty sequence")
    k = min(n_probe, lf)
    probes = np.array([int((i + 0.5) * lf / k) for i in range(k)])
    f = fake_small[probes].astype(np.float32)
    o_all = orig_small.astype(np.float32)
    errs = np.full(lo, np.inf, np.float32)
    need = max(1, int(np.ceil(min_valid * k)))
    for o in range(lo):
        idx = probes + o
        ok = idx < lo
        if ok.sum() < need:
            continue
        errs[o] = float(np.abs(o_all[idx[ok]] - f[ok]).mean())
    best = int(np.argmin(errs))
    far = errs.copy()
    far[max(0, best - 2): best + 3] = np.inf
    second = float(far.min()) if np.isfinite(far).any() else float("inf")
    return OffsetResult(best, float(errs[best]), second, k)
