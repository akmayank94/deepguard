"""Identity clustering, metadata shortcut AUC and manifest validation."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from conftest import model_path
from deepguard.preprocess import audit as AU
from deepguard.preprocess import identity as I
from deepguard.preprocess import manifest as M


def test_umeyama_recovers_similarity():
    src = I.ARCFACE_DST.copy()
    ang, s, t = 0.3, 1.7, np.array([40.0, -12.0])
    R = np.array([[np.cos(ang), -np.sin(ang)], [np.sin(ang), np.cos(ang)]])
    dst = (s * src @ R.T) + t
    m = I.umeyama(src, dst)
    assert np.allclose(m[:, :2], s * R, atol=1e-4) and np.allclose(m[:, 2], t, atol=1e-3)


def test_cluster_and_tune():
    rng = np.random.default_rng(0)
    centers = rng.normal(size=(5, 512))
    truth = np.repeat(np.arange(5), 20)
    emb = centers[truth] + rng.normal(scale=0.6, size=(100, 512))
    t, table = I.tune_threshold(emb, truth)
    s = I.cluster_scores(I.cluster(emb, t), truth)
    assert s["purity"] >= 0.95 and s["completeness"] >= 0.95
    assert I.cluster_scores(I.cluster(emb, 0.99), truth)["n_clusters"] == 100


def test_arcface_same_face_is_close():
    pytest.importorskip("skimage")
    import cv2
    from skimage import data
    from deepguard.preprocess import faces as F

    det = F.make_detector("scrfd_10g", str(model_path("buffalo_l", "det_10g.onnx").parent.parent))
    emb = I.ArcFaceEmbedder(str(model_path("buffalo_l", "w600k_r50.onnx")))
    img = data.astronaut()
    small = cv2.resize(img, (300, 300), interpolation=cv2.INTER_AREA)
    faces = []
    for im in (img, small):
        d = max(det.detect(im), key=lambda x: x.score)
        faces.append(I.norm_crop(im, d.kps))
    noise = np.random.default_rng(0).integers(0, 255, (112, 112, 3)).astype(np.uint8)
    e = emb.embed(faces + [noise])
    assert e.shape == (3, 512)
    assert float(e[0] @ e[1]) > 0.7          # same person, different size
    assert float(e[0] @ e[2]) < 0.3          # not a face


def test_metadata_auc():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame({"is_fake": np.tile([0, 1], n // 2), "group": np.repeat(np.arange(n // 2), 2),
                       "width": rng.choice([640, 854], n), "noise": rng.normal(size=n)})
    ok = AU.metadata_auc(df, ["width", "noise"])
    assert ok["verdict"] == "ok" and ok["auc"] < 0.6
    df.loc[df.is_fake == 1, "width"] = 832                    # planted shortcut
    bad = AU.metadata_auc(df, ["width", "noise"])
    assert bad["verdict"] == "SHORTCUT" and bad["single"]["width"] > 0.9


def _good_manifest():
    rows = []
    for g, split in (("000", "train"), ("001", "test")):
        for meth, label in (("original", "real"), ("Deepfakes", "fake")):
            vid = g if label == "real" else f"{g}_999"
            rows.append(dict(key=f"FFPP/{meth}/c23/{vid}", dataset="FFPP", method=meth,
                             compression="c23", vid=vid, label=label, target=g,
                             source="" if label == "real" else "999", group=g,
                             split_official=split, split_clean=split, identity_cluster=0,
                             fps_changed=False, width_fixed=False, dfd_dup=False, dfd_offset=-1,
                             n_frames=400, n_crops=32, flags="", drop_reason="",
                             raw_sha256="a" * 64, git_commit="abc1234", preprocess_version="1.2.0"))
    return pd.DataFrame(rows)


def test_manifest_validation():
    df = _good_manifest()
    assert M.validate(df) == []
    bad = df.copy()
    bad.loc[1, "split_clean"] = "test"                         # pair group straddles splits
    bad.loc[2, "raw_sha256"] = "xyz"
    bad.loc[3, "n_crops"] = 0
    probs = M.validate(bad)
    assert any("span several splits" in p for p in probs)
    assert any("raw_sha256" in p for p in probs)
    assert any("without crops" in p for p in probs)
    dropped = df.copy()
    dropped.loc[3, ["n_crops", "split_clean", "drop_reason"]] = [0, "", "short_dfd"]
    assert M.validate(dropped) == []                           # dropped rows may lack crops
    assert M.validate(df.drop(columns=["flags"]))[0].startswith("missing columns")
