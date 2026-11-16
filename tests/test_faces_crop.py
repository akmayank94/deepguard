"""Face detection, track linking and cropping."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import model_path
from deepguard.preprocess import crop as C
from deepguard.preprocess import faces as F


def det(x, y, s, score=0.9):
    return F.Detection(np.array([x, y, x + s, y + s], np.float32), score)


def test_iou_nms():
    a = np.array([0, 0, 10, 10], np.float32)
    assert F.iou(a, a) == pytest.approx(1.0)
    assert F.iou(a, np.array([20, 20, 30, 30], np.float32)) == 0
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], np.float32)
    assert sorted(F.nms(boxes, np.array([0.9, 0.8, 0.7]), 0.4)) == [0, 2]


def test_link_tracks_two_faces_and_false_positive():
    frames = {}
    for k, idx in enumerate(range(0, 320, 10)):           # 32 sampled frames
        d = [det(100 + 3 * k, 100, 80), det(500 - 2 * k, 120, 60)]  # two moving faces
        if idx == 150:
            d.append(det(900, 600, 30, 0.6))              # one-frame false detection
        frames[idx] = d[::-1] if k % 2 else d             # detection order must not matter
    tracks = F.link_tracks(frames)
    assert len(tracks) == 3
    kept = F.filter_tracks(tracks, n_frames=32)
    assert sorted(len(t) for t in kept) == [32, 32]
    xs = sorted(t.dets[0].box[0] for t in kept)
    assert xs == [100, 500]


def test_link_tracks_rejects_size_jump():
    frames = {0: [det(100, 100, 80)], 10: [det(105, 100, 200)]}  # 2.5x size: new track
    assert len(F.link_tracks(frames)) == 2


def test_square_box_and_padding():
    rgb = np.full((100, 200, 3), 200, np.uint8)
    crop, (x0, y0, side) = C.crop_square(rgb, np.array([-10, -10, 30, 30], np.float32), out=64)
    assert side == 52 and (x0, y0) == (-16, -16)
    assert crop.shape == (64, 64, 3)
    assert crop[0, 0].max() == 0 and crop[-1, -1].min() == 200   # black outside the frame
    a, _ = C.crop_square(rgb, np.array([50, 20, 90, 70], np.float32))
    b, _ = C.crop_square(rgb, np.array([50, 20, 90, 70], np.float32))
    assert a.shape == (384, 384, 3) and np.array_equal(a, b)    # deterministic


def test_jpeg_roundtrip():
    rng = np.random.default_rng(1)
    img = (np.clip(rng.normal(128, 30, (384, 384, 3)), 0, 255)).astype(np.uint8)
    img = np.ascontiguousarray(np.repeat(np.repeat(img[::8, ::8], 8, 0), 8, 1))
    data = C.encode_jpeg(img)
    back = C.decode_jpeg(data)
    psnr = 10 * np.log10(255 ** 2 / np.mean((back.astype(float) - img) ** 2))
    assert back.shape == img.shape and psnr > 30
    assert C.encode_jpeg(img) == data                            # byte-identical


def _canvas():
    from skimage import data
    img = data.astronaut()
    canvas = np.full((720, 1280, 3), 90, np.uint8)
    canvas[100:612, 50:562] = img
    canvas[100:612, 700:1212] = img[:, ::-1]
    return canvas


@pytest.mark.parametrize("name,rel", [("scrfd_10g", "buffalo_l/det_10g.onnx"),
                                      ("scrfd_500m", "buffalo_sc/det_500m.onnx")])
def test_scrfd_finds_both_faces(name, rel):
    pytest.importorskip("skimage")
    p = model_path(rel)
    d = F.make_detector(name, str(p.parent.parent))
    dets = sorted(d.detect(_canvas()), key=lambda x: x.box[0])
    assert len(dets) == 2 and all(x.score > 0.7 for x in dets)
    a, b = dets
    # the second face is the mirror image of the first: boxes mirror around x = 631
    assert abs((a.box[0] + b.box[2]) - 2 * 631) < 12
    assert abs(a.box[1] - b.box[1]) < 8 and a.kps.shape == (5, 2)
    assert a.kps[0, 0] < a.kps[1, 0]          # image-left eye first


def test_mediapipe_finds_both_faces():
    pytest.importorskip("skimage")
    pytest.importorskip("mediapipe")
    d = F.make_detector("mediapipe")
    dets = d.detect(_canvas())
    d.close()
    assert len(dets) == 2
