"""Right-face selection by pixel difference against the original (D-017.4/5, D-019.3).

A fake and its original share frame indices (FF++), or are aligned by an offset
(DFD, see align.find_offset). For each face track found in the fake, we measure
how much the pixels inside its boxes differ from the original, relative to the
background difference (compression noise). The manipulated face is the track
with the highest ratio. The size of the face never decides.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field

import cv2
import numpy as np

from .faces import Track, iou

LOW_CHANGE_RATIO = 1.5   # best ratio below this -> flag "low_change" (kept, D-017.5)
AMBIGUOUS_SHARE = 0.8    # second-best >= 0.8 x best -> flag "ambiguous"


def _gray(rgb: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.int16)


BG_PAD = 0.15           # background excludes each face box grown by 15% per side


def _clip_box(box: np.ndarray, w: int, h: int, pad: float = 0.0) -> tuple[int, int, int, int]:
    bw, bh = float(box[2] - box[0]), float(box[3] - box[1])
    x1, y1 = max(0, int(box[0] - pad * bw)), max(0, int(box[1] - pad * bh))
    x2, y2 = min(w, int(np.ceil(box[2] + pad * bw))), min(h, int(np.ceil(box[3] + pad * bh)))
    return x1, y1, x2, y2


def background_diff(fake: dict[int, np.ndarray], orig: dict[int, np.ndarray],
                    tracks: list[Track], pad: float = BG_PAD) -> float:
    """Median over frames of the mean |fake - orig| outside every face box.

    Boxes are grown by `pad` per side because blending can reach slightly past
    the detected face (chin, forehead)."""
    vals = []
    for idx in sorted(set(fake) & set(orig)):
        a, b = _gray(fake[idx]), _gray(orig[idx])
        if a.shape != b.shape:
            raise ValueError(f"frame {idx}: fake {a.shape} vs original {b.shape}")
        mask = np.ones(a.shape, bool)
        for t in tracks:
            if idx in t.dets:
                x1, y1, x2, y2 = _clip_box(t.dets[idx].box, a.shape[1], a.shape[0], pad)
                mask[y1:y2, x1:x2] = False
        if mask.any():
            vals.append(float(np.abs(a - b)[mask].mean()))
    return float(np.median(vals)) if vals else float("nan")


def track_diff(fake: dict[int, np.ndarray], orig: dict[int, np.ndarray], t: Track) -> float:
    """Mean |fake - orig| inside the track's boxes, averaged over its frames."""
    vals = []
    for idx, d in t.dets.items():
        if idx not in fake or idx not in orig:
            continue
        a, b = _gray(fake[idx]), _gray(orig[idx])
        x1, y1, x2, y2 = _clip_box(d.box, a.shape[1], a.shape[0])
        if x2 > x1 and y2 > y1:
            vals.append(float(np.abs(a[y1:y2, x1:x2] - b[y1:y2, x1:x2]).mean()))
    return float(np.mean(vals)) if vals else float("nan")


@dataclass
class Selection:
    tid: int | None                       # chosen fake track, None if no track
    ratios: dict[int, float] = field(default_factory=dict)
    bg_diff: float = float("nan")
    flags: list[str] = field(default_factory=list)

    @property
    def best_ratio(self) -> float:
        return self.ratios.get(self.tid, float("nan")) if self.tid is not None else float("nan")


def select_manipulated(fake: dict[int, np.ndarray], orig: dict[int, np.ndarray],
                       tracks: list[Track]) -> Selection:
    """Pick the manipulated track of a fake video (frames must already be aligned)."""
    if not tracks:
        return Selection(None, {}, float("nan"), ["no_track"])
    bg = background_diff(fake, orig, tracks)
    denom = max(bg, 1.0)
    ratios = {t.tid: track_diff(fake, orig, t) / denom for t in tracks}
    ranked = sorted((r, tid) for tid, r in ratios.items() if np.isfinite(r))
    if not ranked:
        return Selection(None, ratios, bg, ["no_overlap"])
    best_r, best = ranked[-1]
    flags = []
    if best_r < LOW_CHANGE_RATIO:
        flags.append("low_change")
    if len(ranked) > 1 and ranked[-2][0] >= AMBIGUOUS_SHARE * best_r:
        flags.append("ambiguous")
    return Selection(best, ratios, bg, flags)


def match_track(t: Track, others: list[Track], min_iou: float = 0.3) -> int | None:
    """Track in `others` that overlaps t most (mean IoU on shared frames)."""
    best, best_v = None, min_iou
    for o in others:
        shared = set(t.dets) & set(o.dets)
        if not shared:
            continue
        v = float(np.mean([iou(t.dets[i].box, o.dets[i].box) for i in shared]))
        if v > best_v:
            best, best_v = o.tid, v
    return best


def vote(choices: list[int | None]) -> tuple[int | None, bool]:
    """Majority vote of the original's main track; returns (track id, unanimous)."""
    valid = [c for c in choices if c is not None]
    if not valid:
        return None, False
    cnt = Counter(valid).most_common()
    top, n = cnt[0]
    if len(cnt) > 1 and cnt[1][1] == n:      # tie -> smallest id, flagged
        top = min(c for c, k in cnt if k == n)
    return top, len(cnt) == 1 and len(valid) == len(choices)
