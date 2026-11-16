"""
deepguard/pipeline.py — inference-time preprocessing + feature extraction.

This mirrors the TRAINING pipeline exactly, so the deployed model sees the
same kind of inputs it was trained and evaluated on:

  scripts/preprocess_final.py      -> face tracking, alignment, crops, windows
  scripts/precompute_features_v2.py-> handcrafted (9369), diff image, PRNU, rPPG
  data/dataset_v2.py               -> raw RGB tensor transform, feature clipping

Differences from training (all deliberate, all documented):
  * Frames are processed as a stream instead of loading the whole video into
    RAM, and at most MAX_FRAMES frames are read (long videos are truncated).
  * Aligned crops are JPEG(q95)-encoded and decoded in memory instead of being
    written to disk; raw crops stay lossless (same as PNG on disk).
  * Clips with fewer than 64 usable raw frames are still analysed (training
    skipped them) but the result is flagged as lower reliability.
"""

from __future__ import annotations

import io
from dataclasses import dataclass, field

import cv2
import numpy as np
from PIL import Image

from .forensics.energy import InterChannelEnergyExtractor
from .forensics.dccm import DCCMExtractor
from .forensics.prnu import extract_prnu_features
from .forensics.rppg import extract_rppg_features

# ── Constants copied from scripts/preprocess_final.py ─────────────────────────
N_ALIGNED = 20
TEMPORAL_GAP = 2
TARGET_MOTION = 0.8
N_RAW = 64
RAW_GAP = 2
OUTPUT_SIZE = 224
JPEG_QUALITY = 95
MAX_CENTER_SHIFT_RATIO = 0.35
SMOOTH_ALPHA = 0.4
WINDOW_STRIDE = 5
MIN_VIDEO_FRAMES = 40          # training rejected clips shorter than this

# ── Deployment-only limits ────────────────────────────────────────────────────
MAX_FRAMES = 450               # ~15-18 s of video; keeps CPU time bounded
MIN_RAW_FOR_ANALYSIS = 20      # below this PRNU/rPPG are meaningless

HC_CLIP = 500.0                # same clip used in precompute + dataset_v2

LEFT_EYE = [33, 133]
RIGHT_EYE = [362, 263]


class PipelineError(Exception):
    """Raised with a user-facing message when a video cannot be analysed."""


@dataclass
class ClipInputs:
    aligned_rgb: list            # N_ALIGNED x (224,224,3) uint8 RGB (after JPEG q95)
    diff_images: np.ndarray      # (N_ALIGNED, 224, 224, 3) float32 in [0,1]
    handcrafted: np.ndarray      # (N_ALIGNED, 9369) float32
    prnu: np.ndarray             # (30,) float32
    rppg: np.ndarray             # (32,) float32
    fps: float
    duration_s: float
    frames_read: int
    aligned_indices: list        # original frame index of each aligned frame
    warnings: list = field(default_factory=list)


# ── MediaPipe ─────────────────────────────────────────────────────────────────
def make_detector():
    import mediapipe as mp
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )


def detect_faces(frame_bgr, detector):
    h, w = frame_bgr.shape[:2]
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    res = detector.process(rgb)
    if res.multi_face_landmarks is None:
        return []
    faces = []
    for lm in res.multi_face_landmarks:
        xs = [p.x for p in lm.landmark]
        ys = [p.y for p in lm.landmark]
        x1 = int(min(xs) * w); x2 = int(max(xs) * w)
        y1 = int(min(ys) * h); y2 = int(max(ys) * h)
        cx = (x1 + x2) // 2; cy = (y1 + y2) // 2
        faces.append((x1, y1, x2, y2, cx, cy, lm))
    return faces


def pick_face(faces, prev_center=None):
    if not faces:
        return None
    if prev_center is None:
        areas = [(f[2] - f[0]) * (f[3] - f[1]) for f in faces]
        return faces[int(np.argmax(areas))]
    dists = [np.hypot(f[4] - prev_center[0], f[5] - prev_center[1]) for f in faces]
    return faces[int(np.argmin(dists))]


def align_and_crop(frame, lm):
    """Rotation-correct around the eye midpoint, crop 4.5x eye distance."""
    h, w = frame.shape[:2]
    lx = np.mean([lm.landmark[i].x for i in LEFT_EYE]) * w
    ly = np.mean([lm.landmark[i].y for i in LEFT_EYE]) * h
    rx = np.mean([lm.landmark[i].x for i in RIGHT_EYE]) * w
    ry = np.mean([lm.landmark[i].y for i in RIGHT_EYE]) * h
    angle = np.degrees(np.arctan2(ry - ly, rx - lx))
    center = (int((lx + rx) / 2), int((ly + ry) / 2))
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(frame, M, (w, h))
    cx = int((lx + rx) / 2); cy = int((ly + ry) / 2)
    eye_dist = np.hypot(lx - rx, ly - ry)
    size = int(eye_dist * 4.5); half = size // 2
    x1 = max(0, cx - half); y1 = max(0, cy - half)
    x2 = min(w, cx + half); y2 = min(h, cy + half)
    crop = aligned[y1:y2, x1:x2]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE))


def raw_crop(frame, face_box):
    """Face box + 20% margin, no rotation (keeps sensor noise intact)."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = face_box
    bw = x2 - x1; bh = y2 - y1
    mx = int(bw * 0.2); my = int(bh * 0.2)
    rx1 = max(0, x1 - mx); ry1 = max(0, y1 - my)
    rx2 = min(w, x2 + mx); ry2 = min(h, y2 + my)
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None, None
    mean_rgb = np.mean(crop.reshape(-1, 3), axis=0).astype(np.float32)[[2, 1, 0]]
    resized = cv2.resize(crop, (OUTPUT_SIZE, OUTPUT_SIZE), interpolation=cv2.INTER_AREA)
    return resized, mean_rgb


# ── Window selection (same logic as training, with cached optical flow) ──────
def _flow_mag(a_bgr, b_bgr):
    a = cv2.cvtColor(a_bgr, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(b_bgr, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    return float(np.mean(np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)))


def select_aligned_window(face_frames):
    """Returns the list of indices (into face_frames) of the best window."""
    needed = N_ALIGNED * TEMPORAL_GAP
    cache = {}
    best, best_diff = None, float("inf")
    for start in range(0, len(face_frames) - needed + 1, WINDOW_STRIDE):
        idx = [start + i * TEMPORAL_GAP for i in range(N_ALIGNED)
               if start + i * TEMPORAL_GAP < len(face_frames)]
        if len(idx) < N_ALIGNED:
            continue
        mags = []
        for a, b in zip(idx[:-1], idx[1:]):
            if (a, b) not in cache:
                cache[(a, b)] = _flow_mag(face_frames[a], face_frames[b])
            mags.append(cache[(a, b)])
        diff = abs(float(np.mean(mags)) - TARGET_MOTION)
        if diff < best_diff:
            best_diff, best = diff, idx
    return best


def select_raw_window(raw_items):
    needed = N_RAW * RAW_GAP
    if len(raw_items) < needed:
        indices = np.linspace(0, len(raw_items) - 1, N_RAW, dtype=int)
        return [raw_items[i] for i in indices]
    mid = len(raw_items) // 2
    start = max(0, mid - needed // 2)
    seq = [raw_items[start + i * RAW_GAP] for i in range(N_RAW)
           if start + i * RAW_GAP < len(raw_items)]
    while len(seq) < N_RAW:
        seq.append(raw_items[-1])
    return seq[:N_RAW]


def _jpeg_roundtrip_rgb(bgr):
    """Training saved aligned crops as JPEG q95 and re-read them with PIL."""
    ok, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    if not ok:
        raise PipelineError("Internal error while encoding a face crop.")
    return np.array(Image.open(io.BytesIO(buf.tobytes())).convert("RGB"))


# ── Public entry point ────────────────────────────────────────────────────────
_energy = InterChannelEnergyExtractor()
_dccm = DCCMExtractor()


def extract_handcrafted_and_diff(img_rgb):
    e = _energy.extract(img_rgb)
    d = _dccm.extract(img_rgb)
    hc = np.concatenate([e["feature_vector"].astype(np.float32),
                         d["feature_vector"].astype(np.float32)])
    hc = np.clip(hc, -HC_CLIP, HC_CLIP).astype(np.float32)
    return hc, e["diff_image"].astype(np.float32)


def prepare_clip(video_path: str, progress=None) -> ClipInputs:
    def _p(frac, msg):
        if progress is not None:
            progress(frac, desc=msg)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise PipelineError("Could not open this video. Try MP4 (H.264).")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    if not np.isfinite(fps) or fps <= 1:
        fps = 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    detector = make_detector()
    prev_center = prev_size = None
    aligned_frames, aligned_src = [], []
    raw_items = []
    frames_read = 0
    warnings = []

    _p(0.05, "Tracking face")
    try:
        while frames_read < MAX_FRAMES:
            ret, frame = cap.read()
            if not ret:
                break
            fi = frames_read
            frames_read += 1
            if frames_read % 30 == 0:
                denom = min(total, MAX_FRAMES) if total > 0 else MAX_FRAMES
                _p(0.05 + 0.45 * min(1.0, frames_read / denom), "Tracking face")

            face = pick_face(detect_faces(frame, detector), prev_center)
            if face is None:
                continue
            x1, y1, x2, y2, cx, cy, lm = face
            size = max(x2 - x1, y2 - y1)
            if prev_center is not None:
                max_shift = prev_size * MAX_CENTER_SHIFT_RATIO
                shift = np.hypot(cx - prev_center[0], cy - prev_center[1])
                if shift > max_shift:
                    ratio = max_shift / shift
                    cx = int(prev_center[0] + (cx - prev_center[0]) * ratio)
                    cy = int(prev_center[1] + (cy - prev_center[1]) * ratio)
                size = int(SMOOTH_ALPHA * size + (1 - SMOOTH_ALPHA) * prev_size)
            prev_center = (cx, cy); prev_size = size

            a = align_and_crop(frame, lm)
            if a is not None:
                aligned_frames.append(a)
                aligned_src.append(fi)
            rc, mean_rgb = raw_crop(frame, (x1, y1, x2, y2))
            if rc is not None:
                raw_items.append((rc, mean_rgb))
    finally:
        cap.release()
        detector.close()

    if total > MAX_FRAMES and frames_read >= MAX_FRAMES:
        warnings.append(f"Only the first {MAX_FRAMES / fps:.0f}s of the video were analysed.")
    if frames_read < MIN_VIDEO_FRAMES:
        raise PipelineError(
            f"Video is too short ({frames_read} frames). Please upload at least ~2 seconds.")
    if len(aligned_frames) < N_ALIGNED * TEMPORAL_GAP:
        raise PipelineError(
            "Could not track a clear face in enough frames. Use a video where one "
            "face is visible, front-facing and well lit for at least 2 seconds.")
    if len(raw_items) < MIN_RAW_FOR_ANALYSIS:
        raise PipelineError("Not enough face frames for noise analysis.")
    if len(raw_items) < N_RAW:
        warnings.append("Short clip: fewer than 64 face frames, so the PRNU and rPPG "
                        "branches are less reliable than in evaluation.")

    _p(0.55, "Selecting stable frame window")
    win = select_aligned_window(aligned_frames)
    if win is None:
        raise PipelineError("Could not find a stable 20-frame window in this video.")

    best_raw = select_raw_window(raw_items)
    raw_bgr = [r for r, _ in best_raw]
    rppg_signal = np.array([m for _, m in best_raw], dtype=np.float32)

    _p(0.62, "Extracting spectral + contourlet features")
    aligned_rgb, hcs, diffs = [], [], []
    for k, i in enumerate(win):
        rgb = _jpeg_roundtrip_rgb(aligned_frames[i])
        hc, di = extract_handcrafted_and_diff(rgb)
        aligned_rgb.append(rgb); hcs.append(hc); diffs.append(di)
        _p(0.62 + 0.25 * (k + 1) / len(win), "Extracting spectral + contourlet features")

    _p(0.88, "Extracting PRNU + rPPG signals")
    prnu = extract_prnu_features(raw_bgr, n_frames=20)
    rppg = extract_rppg_features(rppg_signal, fps=fps / RAW_GAP)

    return ClipInputs(
        aligned_rgb=aligned_rgb,
        diff_images=np.stack(diffs).astype(np.float32),
        handcrafted=np.stack(hcs).astype(np.float32),
        prnu=prnu.astype(np.float32),
        rppg=rppg.astype(np.float32),
        fps=float(fps),
        duration_s=float(total / fps) if total > 0 else float(frames_read / fps),
        frames_read=frames_read,
        aligned_indices=[aligned_src[i] for i in win],
        warnings=warnings,
    )
