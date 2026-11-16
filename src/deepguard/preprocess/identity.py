"""Identity embeddings and clustering (D-018.3, D-019.8).

ArcFace (InsightFace ``w600k_r50.onnx``) embeds the main face of each video; videos
whose embeddings are close enough (cosine >= t) are linked, and connected groups
become identity clusters. t is tuned on DFD, where the 28 actors are known, and the
same t is then used on FF++ and the other datasets.
"""
from __future__ import annotations

import cv2
import numpy as np

# InsightFace 112x112 ArcFace template (left eye, right eye, nose, mouth corners).
ARCFACE_DST = np.array([[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
                        [41.5493, 92.3655], [70.7299, 92.2041]], np.float32)


def umeyama(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Least-squares similarity transform (rotation, uniform scale, shift): 2x3."""
    src = np.asarray(src, np.float64)
    dst = np.asarray(dst, np.float64)
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s_c, d_c = src - mu_s, dst - mu_d
    cov = d_c.T @ s_c / len(src)
    u, s, vt = np.linalg.svd(cov)
    d = np.ones(2)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        d[-1] = -1
    r = u @ np.diag(d) @ vt
    var = (s_c ** 2).sum() / len(src)
    scale = (s * d).sum() / var
    t = mu_d - scale * r @ mu_s
    return np.hstack([scale * r, t[:, None]]).astype(np.float32)


def norm_crop(rgb: np.ndarray, kps: np.ndarray, size: int = 112) -> np.ndarray:
    """Align a face to the ArcFace template using its 5 landmarks."""
    m = umeyama(kps, ARCFACE_DST * (size / 112.0))
    return cv2.warpAffine(rgb, m, (size, size), borderValue=0.0)


class ArcFaceEmbedder:
    name = "arcface_w600k_r50"

    def __init__(self, model_path: str, providers: list[str] | None = None):
        import onnxruntime as ort

        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in avail]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.input_name = self.session.get_inputs()[0].name

    def embed(self, aligned_rgb: list[np.ndarray], batch: int = 64) -> np.ndarray:
        """L2-normalised (n, 512) embeddings of 112x112 aligned RGB faces."""
        outs = []
        for i in range(0, len(aligned_rgb), batch):
            x = np.stack(aligned_rgb[i:i + batch]).astype(np.float32)
            x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
            outs.append(self.session.run(None, {self.input_name: x})[0])
        e = np.concatenate(outs).astype(np.float32)
        return e / np.linalg.norm(e, axis=1, keepdims=True).clip(1e-9)


def cluster(emb: np.ndarray, thresh: float) -> np.ndarray:
    """Connected components of the graph 'cosine >= thresh' (union-find). Labels 0..k-1."""
    e = emb / np.linalg.norm(emb, axis=1, keepdims=True).clip(1e-9)
    n = len(e)
    parent = np.arange(n)

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for start in range(0, n, 2048):
        sims = e[start:start + 2048] @ e.T
        rows, cols = np.nonzero(sims >= thresh)
        for i, j in zip(rows + start, cols):
            if i < j:
                ri, rj = find(i), find(j)
                if ri != rj:
                    parent[max(ri, rj)] = min(ri, rj)
    roots = np.array([find(i) for i in range(n)])
    _, labels = np.unique(roots, return_inverse=True)
    return labels


def cluster_scores(pred: np.ndarray, truth: np.ndarray) -> dict:
    """Purity (each cluster mostly one identity) and completeness (each identity in one cluster)."""
    pred, truth = np.asarray(pred), np.asarray(truth)
    n = len(pred)
    purity = sum(np.unique(truth[pred == c], return_counts=True)[1].max()
                 for c in np.unique(pred)) / n
    complete = sum(np.unique(pred[truth == c], return_counts=True)[1].max()
                   for c in np.unique(truth)) / n
    return {"purity": float(purity), "completeness": float(complete),
            "n_clusters": int(len(np.unique(pred))), "n_identities": int(len(np.unique(truth)))}


def tune_threshold(emb: np.ndarray, truth: np.ndarray, grid=None, min_purity: float = 0.95):
    """Scan thresholds; pick the lowest t whose purity >= min_purity (most merging
    that is still clean), because leakage control prefers merging too much over
    too little. Returns (t, table)."""
    if grid is None:
        grid = np.round(np.arange(0.20, 0.80, 0.025), 3)
    table = []
    for t in grid:
        s = cluster_scores(cluster(emb, float(t)), truth)
        s["thresh"] = float(t)
        table.append(s)
    ok = [r for r in table if r["purity"] >= min_purity]
    best = min(ok, key=lambda r: r["thresh"]) if ok else max(table, key=lambda r: r["purity"])
    return best["thresh"], table
