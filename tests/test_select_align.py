"""Right-face selection, width-transform test and DFD offset matching on synthetic data."""
from __future__ import annotations

import cv2
import numpy as np

from deepguard.preprocess import align as A
from deepguard.preprocess import faces as F
from deepguard.preprocess import select as S

RNG = np.random.default_rng(7)


def texture(h, w, seed=0):
    r = np.random.default_rng(seed)
    small = r.integers(0, 255, (h // 8 + 1, w // 8 + 1, 3)).astype(np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_CUBIC)[:h, :w]


def noisy(img, sd=2.0):
    return np.clip(img + RNG.normal(0, sd, img.shape), 0, 255).astype(np.uint8)


def make_pair(edit_box, strength=25.0, n=8, h=240, w=320):
    orig, fake = {}, {}
    for i in range(n):
        base = texture(h, w, seed=i)
        o = noisy(base)
        f = noisy(base).astype(np.float32)
        x1, y1, x2, y2 = edit_box
        f[y1:y2, x1:x2] += strength * np.sin(np.arange(x2 - x1) / 3.0)[None, :, None]
        orig[i], fake[i] = o, np.clip(f, 0, 255).astype(np.uint8)
    return orig, fake


def tracks_for(boxes, frames):
    return [F.Track(t, {i: F.Detection(np.array(b, np.float32), 0.9) for i in frames})
            for t, b in enumerate(boxes)]


def test_selects_edited_face_not_largest():
    big, small = (20, 40, 140, 180), (220, 60, 270, 120)     # small face is the fake one
    orig, fake = make_pair(small)
    sel = S.select_manipulated(fake, orig, tracks_for([big, small], orig))
    assert sel.tid == 1 and "ambiguous" not in sel.flags
    assert sel.best_ratio > 3 and sel.bg_diff < 4


def test_small_edit_is_kept_with_flag():
    face = (100, 60, 220, 200)
    orig, fake = make_pair((140, 160, 180, 175), strength=4.0)  # tiny mouth-only change
    sel = S.select_manipulated(fake, orig, tracks_for([face], orig))
    assert sel.tid == 0 and "low_change" in sel.flags


def test_ambiguous_when_both_faces_changed():
    a, b = (20, 40, 120, 160), (200, 40, 300, 160)
    orig, fake = make_pair(a)
    for i in fake:                       # edit face b equally
        f = fake[i].astype(np.float32)
        f[40:160, 200:300] += 25 * np.sin(np.arange(100) / 3.0)[None, :, None]
        fake[i] = np.clip(f, 0, 255).astype(np.uint8)
    sel = S.select_manipulated(fake, orig, tracks_for([a, b], orig))
    assert "ambiguous" in sel.flags


def test_no_track():
    assert S.select_manipulated({}, {}, []).flags == ["no_track"]


def test_vote_and_match():
    assert S.vote([2, 2, 1, 2, 2]) == (2, False)
    assert S.vote([0, 0, 0]) == (0, True)
    assert S.vote([None, None]) == (None, False)
    t = tracks_for([(10, 10, 50, 50)], [0, 1])[0]
    others = tracks_for([(300, 300, 340, 340), (12, 11, 52, 51)], [0, 1])
    assert S.match_track(t, others) == 1


def _width_case(mode, offset=11):
    h, w = 240, 854
    o = texture(h, w, seed=3)
    if mode == "crop":
        f = o[:, offset:offset + 832].copy()
    else:
        f = cv2.resize(o, (832, h), interpolation=cv2.INTER_AREA)
    f = noisy(f)
    box = np.array([300, 60, 420, 200], np.float32)
    f[60:200, 300:420] = 255 - f[60:200, 300:420]          # manipulated face area
    return o, f, [box]


def test_width_detects_crop_offset():
    o, f, boxes = _width_case("crop", 11)
    r = A.width_candidates(o, f, boxes)
    assert r["best"][:2] == ("crop", 11)
    assert A.crop_offset_rule(854, 11) == "center" and A.crop_offset_rule(854, 0) == "left"


def test_width_detects_rescale():
    o, f, boxes = _width_case("rescale")
    assert A.width_candidates(o, f, boxes)["best"][0] == "rescale"


def test_find_offset():
    seq = np.stack([cv2.resize(texture(72, 128, seed=i)[:, :, 0], (64, 36),
                               interpolation=cv2.INTER_AREA) for i in range(120)])
    seq = (0.5 * seq + 0.5 * np.roll(seq, 1, axis=0)).astype(np.uint8)   # smooth in time
    fake = np.clip(seq[23:95] + RNG.normal(0, 3, seq[23:95].shape), 0, 255).astype(np.uint8)
    fake[:, 10:20, 25:35] = 0                                             # changed face
    r = A.find_offset(fake, seq)
    assert r.offset == 23 and r.margin > 1.5


def test_find_offset_fake_longer_than_rest():
    seq = np.stack([texture(36, 64, seed=i)[:, :, 0] for i in range(50)])
    r = A.find_offset(seq[30:50], seq)        # fake runs to the original's last frame
    assert r.offset == 30 and r.err == 0
