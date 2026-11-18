"""Face detection and track linking (D-019.3, D-019.6).

Two detectors behind one interface, so Phase 1.2 can choose by measurement:

* ``ScrfdDetector``  - SCRFD (InsightFace ``det_10g.onnx`` / ``det_500m.onnx``) via
  onnxruntime. Gives a box, a score and 5 landmarks (eyes, nose, mouth corners).
  The decoding below re-implements InsightFace's ``scrfd.py`` without the package.
* ``MediaPipeDetector`` - MediaPipe face detection (full-range model). Gives a box,
  a score and eyes/nose/mouth-centre points; mouth corners are not available.

Both take RGB uint8 frames and return pixel coordinates in that frame.
InsightFace models are licensed for non-commercial research only (D-018.9).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np


@dataclass
class Detection:
    box: np.ndarray            # x1, y1, x2, y2 (float32, pixels)
    score: float
    kps: np.ndarray | None = None  # 5 x 2 (float32) or None

    @property
    def size(self) -> float:
        return float(max(self.box[2] - self.box[0], self.box[3] - self.box[1]))

    @property
    def center(self) -> np.ndarray:
        return np.array([(self.box[0] + self.box[2]) / 2, (self.box[1] + self.box[3]) / 2],
                        np.float32)


def iou(a: np.ndarray, b: np.ndarray) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / ua) if ua > 0 else 0.0


def nms(boxes: np.ndarray, scores: np.ndarray, thresh: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = int(order[0])
        keep.append(i)
        rest = order[1:]
        if not rest.size:
            break
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        area = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        ovr = inter / (area[i] + area[rest] - inter + 1e-9)
        order = rest[ovr <= thresh]
    return keep


class ScrfdDetector:
    """SCRFD face detector on onnxruntime (GPU if available, else CPU)."""

    name = "scrfd"

    def __init__(self, model_path: str, det_size: int = 640, score_thresh: float = 0.5,
                 nms_thresh: float = 0.4, providers: list[str] | None = None):
        import onnxruntime as ort

        if providers is None:
            avail = ort.get_available_providers()
            providers = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider")
                         if p in avail]
        so = ort.SessionOptions()
        so.log_severity_level = 3
        self.session = ort.InferenceSession(str(model_path), so, providers=providers)
        self.providers = self.session.get_providers()
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [o.name for o in self.session.get_outputs()]
        self.det_size = det_size
        self.score_thresh = score_thresh
        self.nms_thresh = nms_thresh
        n = len(self.output_names)
        if n not in (6, 9):
            raise ValueError(f"unexpected SCRFD output count {n}")
        self.fmc = 3
        self.strides = [8, 16, 32]
        self.use_kps = n == 9
        self.num_anchors = 2
        self._centers: dict = {}

    def _anchor_centers(self, h: int, w: int, stride: int) -> np.ndarray:
        key = (h, w, stride)
        if key not in self._centers:
            c = np.stack(np.mgrid[:h, :w][::-1], axis=-1).astype(np.float32)
            c = (c * stride).reshape(-1, 2)
            c = np.stack([c] * self.num_anchors, axis=1).reshape(-1, 2)
            self._centers[key] = c
        return self._centers[key]

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        h0, w0 = rgb.shape[:2]
        s = self.det_size
        scale = min(s / h0, s / w0)
        nh, nw = int(round(h0 * scale)), int(round(w0 * scale))
        canvas = np.zeros((s, s, 3), np.uint8)
        canvas[:nh, :nw] = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
        # InsightFace feeds RGB, normalised as (x - 127.5) / 128.
        blob = ((canvas.astype(np.float32) - 127.5) / 128.0).transpose(2, 0, 1)[None]
        outs = self.session.run(self.output_names, {self.input_name: blob})
        boxes_l, scores_l, kps_l = [], [], []
        for i, stride in enumerate(self.strides):
            scores = outs[i].reshape(-1)
            bbox = outs[i + self.fmc].reshape(-1, 4) * stride
            centers = self._anchor_centers(s // stride, s // stride, stride)
            pos = np.where(scores >= self.score_thresh)[0]
            if not pos.size:
                continue
            c = centers[pos]
            b = bbox[pos]
            boxes_l.append(np.stack([c[:, 0] - b[:, 0], c[:, 1] - b[:, 1],
                                     c[:, 0] + b[:, 2], c[:, 1] + b[:, 3]], axis=-1))
            scores_l.append(scores[pos])
            if self.use_kps:
                k = outs[i + 2 * self.fmc].reshape(-1, 10)[pos] * stride
                k = k.reshape(-1, 5, 2) + c[:, None, :]
                kps_l.append(k)
        if not boxes_l:
            return []
        boxes = np.concatenate(boxes_l) / scale
        scores = np.concatenate(scores_l)
        kps = np.concatenate(kps_l) / scale if kps_l else None
        keep = nms(boxes, scores, self.nms_thresh)
        dets = []
        for i in keep:
            b = boxes[i].astype(np.float32)
            b[[0, 2]] = np.clip(b[[0, 2]], 0, w0)
            b[[1, 3]] = np.clip(b[[1, 3]], 0, h0)
            if b[2] - b[0] < 2 or b[3] - b[1] < 2:
                continue
            dets.append(Detection(b, float(scores[i]),
                                  kps[i].astype(np.float32) if kps is not None else None))
        return dets


MEDIAPIPE_FULL_RANGE = "face_detection_full_range_sparse.tflite"
MEDIAPIPE_FULL_RANGE_SHA256 = "2c3728e6da56f21e21a320433396fb06d40d9088f2247c05e5635a688d45dfe1"


class MediaPipeDetector:
    """MediaPipe BlazeFace full-range detector.

    Two APIs, same model: MediaPipe <= 0.10.2x has the legacy ``solutions`` API
    (model_selection=1 = ``face_detection_full_range_sparse.tflite``, bundled);
    newer releases (0.10.30+, 1.x) only have the Tasks API, which needs the model
    file passed as ``model_path``. Passing ``model_path`` forces the Tasks API.
    """

    name = "mediapipe"

    def __init__(self, score_thresh: float = 0.5, model_path: str | None = None):
        import mediapipe as mp

        self._mp = mp
        if model_path is None and hasattr(mp, "solutions"):
            self.api = "solutions"
            self._fd = mp.solutions.face_detection.FaceDetection(
                model_selection=1, min_detection_confidence=score_thresh)
        else:
            if model_path is None:
                raise ValueError(f"MediaPipe {mp.__version__} has no legacy API: pass "
                                 f"model_path to {MEDIAPIPE_FULL_RANGE}")
            from mediapipe.tasks import python as mpt
            from mediapipe.tasks.python import vision

            self.api = "tasks"
            self._fd = vision.FaceDetector.create_from_options(vision.FaceDetectorOptions(
                base_options=mpt.BaseOptions(model_asset_path=str(model_path)),
                min_detection_confidence=score_thresh))

    @staticmethod
    def _finish(box, score, pts, w, h) -> Detection | None:
        box[[0, 2]] = np.clip(box[[0, 2]], 0, w)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, h)
        if box[2] - box[0] < 2 or box[3] - box[1] < 2:
            return None
        # MediaPipe order: right eye, left eye, nose tip, mouth centre, ears.
        # Map to InsightFace order (left-of-image eye first) with the mouth
        # centre repeated for both mouth corners (no corners in BlazeFace).
        kps = np.stack([pts[0], pts[1], pts[2], pts[3], pts[3]]).astype(np.float32)
        return Detection(box.astype(np.float32), float(score), kps)

    def detect(self, rgb: np.ndarray) -> list[Detection]:
        h, w = rgb.shape[:2]
        out: list[Detection] = []
        if self.api == "solutions":
            res = self._fd.process(rgb)
            for d in res.detections or []:
                bb = d.location_data.relative_bounding_box
                box = np.array([bb.xmin * w, bb.ymin * h, (bb.xmin + bb.width) * w,
                                (bb.ymin + bb.height) * h], np.float32)
                pts = np.array([[p.x * w, p.y * h] for p in d.location_data.relative_keypoints[:4]])
                det = self._finish(box, d.score[0], pts, w, h)
                if det is not None:
                    out.append(det)
            return out
        img = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb))
        for d in self._fd.detect(img).detections:
            bb = d.bounding_box
            box = np.array([bb.origin_x, bb.origin_y, bb.origin_x + bb.width,
                            bb.origin_y + bb.height], np.float32)
            pts = np.array([[p.x * w, p.y * h] for p in d.keypoints[:4]])
            det = self._finish(box, d.categories[0].score, pts, w, h)
            if det is not None:
                out.append(det)
        return out

    def close(self):
        self._fd.close()


def make_detector(name: str, models_dir: str | None = None, **kw):
    """Factory used by the notebooks: 'scrfd_10g', 'scrfd_500m' or 'mediapipe'."""
    from pathlib import Path

    if name == "mediapipe":
        return MediaPipeDetector(**kw)
    files = {"scrfd_10g": "buffalo_l/det_10g.onnx", "scrfd_500m": "buffalo_sc/det_500m.onnx"}
    if name not in files:
        raise ValueError(f"unknown detector {name!r}")
    if models_dir is None:
        raise ValueError("models_dir is required for SCRFD")
    return ScrfdDetector(str(Path(models_dir) / files[name]), **kw)


# ---------------------------------------------------------------- tracks
@dataclass
class Track:
    """One face followed across the sampled frames of a video."""
    tid: int
    dets: dict[int, Detection] = field(default_factory=dict)  # frame idx -> detection

    @property
    def last(self) -> Detection:
        return self.dets[max(self.dets)]

    def mean_size(self) -> float:
        return float(np.mean([d.size for d in self.dets.values()]))

    def mean_score(self) -> float:
        return float(np.mean([d.score for d in self.dets.values()]))

    def __len__(self) -> int:
        return len(self.dets)


def _match_cost(t: Track, d: Detection, iou_min: float, dist_max: float,
                size_ratio_max: float) -> float | None:
    """Cost of attaching detection d to track t, or None if not allowed."""
    last = t.last
    ratio = max(d.size, last.size) / max(1e-6, min(d.size, last.size))
    if ratio > size_ratio_max:
        return None
    ov = iou(last.box, d.box)
    dist = float(np.linalg.norm(d.center - last.center)) / max(last.size, d.size)
    if ov >= iou_min or dist <= dist_max:
        return dist - ov
    return None


def link_tracks(frames: dict[int, list[Detection]], iou_min: float = 0.3,
                dist_max: float = 0.5, size_ratio_max: float = 1.6) -> list[Track]:
    """Greedy linking of per-frame detections into tracks.

    Sampled frames can be ~15 frames apart, so a detection joins a track when it
    overlaps the track's last box (IoU >= iou_min) or its centre moved less than
    dist_max x the face size, and the size changed by less than size_ratio_max.
    Each frame's detections are assigned to tracks by lowest cost first; a track
    takes at most one detection per frame. Unmatched detections start new tracks.
    """
    tracks: list[Track] = []
    for idx in sorted(frames):
        dets = frames[idx]
        pairs = []
        for di, d in enumerate(dets):
            for ti, t in enumerate(tracks):
                c = _match_cost(t, d, iou_min, dist_max, size_ratio_max)
                if c is not None:
                    pairs.append((c, di, ti))
        pairs.sort()
        used_d, used_t = set(), set()
        for c, di, ti in pairs:
            if di in used_d or ti in used_t:
                continue
            tracks[ti].dets[idx] = dets[di]
            used_d.add(di)
            used_t.add(ti)
        for di, d in enumerate(dets):
            if di not in used_d:
                tracks.append(Track(len(tracks), {idx: d}))
    return tracks


def filter_tracks(tracks: list[Track], n_frames: int, min_share: float = 0.5) -> list[Track]:
    """Drop short tracks (false detections such as logos): D-019.3."""
    need = int(np.ceil(min_share * n_frames))
    return [t for t in tracks if len(t) >= need]
