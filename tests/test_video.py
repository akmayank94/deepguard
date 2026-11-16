"""D-019.1 colour test and decoding behaviour of deepguard.preprocess.video."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import frame_index_video, write_yuv_video
from deepguard.preprocess import video as V


def analytic_rgb(Y, Cb, Cr, matrix, rng):
    kr, kb = V.MATRICES[matrix]
    kg = 1 - kr - kb
    if rng == "limited":
        y, cb, cr = (Y - 16) / 219, (Cb - 128) / 224, (Cr - 128) / 224
    else:
        y, cb, cr = Y / 255, (Cb - 128) / 255, (Cr - 128) / 255
    r = y + 2 * (1 - kr) * cr
    b = y + 2 * (1 - kb) * cb
    g = (y - kr * r - kb * b) / kg
    return np.clip(np.rint(np.array([r, g, b]) * 255), 0, 255)


def patch_frame(h, w, values, block=16, sub=2):
    """Flat blocks with known (Y, Cb, Cr); chroma is constant inside each block.

    sub=2 makes 4:2:0 chroma planes, sub=1 makes 4:4:4 (the FF++ format)."""
    y = np.zeros((h, w), np.uint8)
    u = np.zeros((h // sub, w // sub), np.uint8)
    v = np.zeros((h // sub, w // sub), np.uint8)
    cols, b = w // block, block // sub
    for k, (Y, Cb, Cr) in enumerate(values):
        r, c = divmod(k, cols)
        y[r * block:(r + 1) * block, c * block:(c + 1) * block] = Y
        u[r * b:(r + 1) * b, c * b:(c + 1) * b] = Cb
        v[r * b:(r + 1) * b, c * b:(c + 1) * b] = Cr
    return y, u, v


RNG = np.random.default_rng(0)
VALUES = [(int(a), int(b), int(c)) for a, b, c in
          zip(RNG.integers(16, 236, 40), RNG.integers(16, 241, 40), RNG.integers(16, 241, 40))]


@pytest.mark.parametrize("h,w,cs,cr,fmt,expect", [
    (480, 640, None, None, "yuv420p", ("bt601", "limited")),    # untagged SD -> BT.601
    (720, 1280, None, None, "yuv420p", ("bt709", "limited")),   # untagged HD -> BT.709
    (480, 640, 1, 1, "yuv420p", ("bt709", "limited")),          # tag wins over size
    (720, 1280, 6, 1, "yuv420p", ("bt601", "limited")),
    (480, 640, 1, 2, "yuv420p", ("bt709", "full")),
    (480, 640, None, None, "yuv444p", ("bt601", "limited")),    # FF++ format (4:4:4)
])
def test_colour_matches_analytic(tmp_path, h, w, cs, cr, fmt, expect):
    vals = VALUES[: (h // 16) * (w // 16)]
    y, u, v = patch_frame(h, w, vals, sub=1 if fmt == "yuv444p" else 2)
    p = write_yuv_video(tmp_path / "c.mp4", [(y, u, v)] * 2, colorspace=cs, color_range=cr,
                        pix_fmt=fmt)
    info = V.probe(str(p))
    # libx264 reports full-range 4:2:0 as "yuvj420p"
    assert info.pix_fmt in (fmt, "yuvj" + fmt[3:])
    assert V.choose_conversion(info.height, info.pix_fmt, info.colorspace_tag,
                               info.color_range_tag) == expect
    rgb = V.decode_frames(str(p), [0])[0]
    worst = 0
    for k, (Y, Cb, Cr) in enumerate(vals):
        r, c = divmod(k, w // 16)
        # centre pixel of the block: chroma interpolation cannot touch it
        got = rgb[r * 16 + 8, c * 16 + 8].astype(int)
        want = analytic_rgb(Y, Cb, Cr, *expect)
        worst = max(worst, int(np.abs(got - want).max()))
    assert worst <= 1, f"max channel error {worst}"


def test_choose_conversion_rules():
    assert V.choose_conversion(1080, "yuv420p", 2, 0) == ("bt709", "limited")
    assert V.choose_conversion(360, "yuvj420p", 2, 0) == ("bt601", "full")
    assert V.choose_conversion(360, "yuv420p", 9, 2) == ("bt2020", "full")


def test_index_order_and_count(tmp_path):
    p = frame_index_video(tmp_path / "i.mp4", 40)
    assert V.count_frames(str(p)) == 40
    got = V.decode_frames(str(p), [0, 5, 39, 99])
    assert sorted(got) == [0, 5, 39]  # 99 does not exist: absent, not an error
    for i in (0, 5, 39):
        assert abs(int(got[i][10, 10, 0]) - analytic_rgb(16 + (i * 7) % 200, 128, 128,
                                                         "bt601", "limited")[0]) <= 1


def test_sample_indices():
    assert V.sample_indices(31, 32) is None
    idx = V.sample_indices(320, 32)
    assert len(idx) == 32 and idx[0] == 5 and idx[-1] == 315
    assert idx == sorted(set(idx))
    assert V.sample_indices(32, 32) == list(range(32))


def test_gray_small(tmp_path):
    p = frame_index_video(tmp_path / "g.mp4", 12)
    g = list(V.iter_gray_small(str(p), (16, 9)))
    assert len(g) == 12 and g[3].shape == (9, 16)
    assert int(g[3][4, 4]) == 16 + 21


def test_normalize_width():
    fr = np.arange(10 * 854 * 3, dtype=np.uint32).reshape(10, 854, 3).astype(np.uint8)
    assert V.normalize_width(fr, None).shape == (10, 854, 3)
    c = V.normalize_width(fr, "crop", 11)       # 854 -> 832, as in FF++
    assert c.shape == (10, 832, 3) and np.array_equal(c, fr[:, 11:843])
    assert V.normalize_width(fr, "rescale").shape == (10, 832, 3)
    assert V.normalize_width(fr[:, :720], "crop", 16).shape == (10, 704, 3)  # 720 -> 704
    ok = np.zeros((10, 832, 3), np.uint8)
    assert V.normalize_width(ok, "crop", 5) is ok
    with pytest.raises(ValueError):
        V.normalize_width(fr, "crop", 23)
