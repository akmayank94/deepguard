"""
scripts/preprocess_final.py — FINAL PREPROCESSING (DO NOT RUN AGAIN AFTER THIS)

Saves per clip:
  frames_aligned/  — 20 aligned 224x224 JPG face crops (for EfficientNet)
  frames_raw/      — 64 unaligned 224x224 PNG face patches (for PRNU)
  rppg.npy         — (64, 3) mean RGB per frame before resize (for rPPG)
  meta.json        — fps, source_video, frame_indices used

Design decisions:
  - Aligned frames:  rotation-corrected, landmark-cropped, 224x224, JPEG quality 95
  - Raw frames:      NO alignment, face bbox crop only, 224x224, PNG (lossless)
  - rPPG:           mean R,G,B of the face region (before resize) across 64 frames
  - TEMPORAL_GAP=2  for aligned frames (same as original, model already trained on this)
  - Raw/rPPG:       64 consecutive frames, every 2nd original frame (effective 12.5fps)
  - Video-level split will be done later in dataset_ff.py (same as before)

Usage:
    python scripts/preprocess_final.py
    python scripts/preprocess_final.py --test_mode    (5 videos per type)
    python scripts/preprocess_final.py --resume       (skip already-done clips)
"""

import cv2
import numpy as np
import json
import argparse
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_ROOT   = Path("D:/DO NOT TOUCH/deepfake_microexpression/data/FaceForensics++_C23")
OUTPUT_ROOT = Path("D:/DO NOT TOUCH/deepfake_microexpression/data/FF++C23_preprocessed")

FAKE_TYPES  = ["Deepfakes", "Face2Face", "FaceShifter", "FaceSwap", "NeuralTextures"]

# Aligned frames (for CNN)
N_ALIGNED    = 20       # frames saved for CNN branches
TEMPORAL_GAP = 2        # every 2nd face frame, same as original training data
TARGET_MOTION = 0.8     # same motion target as original

# Raw frames (for PRNU + rPPG)
N_RAW        = 64       # frames saved for PRNU/rPPG — covers ~5s at effective 12.5fps
RAW_GAP      = 2        # every 2nd original frame (effective fps = original_fps / 2)

OUTPUT_SIZE  = 224      # both aligned and raw saved at 224x224
JPEG_QUALITY = 95       # high quality JPG for aligned (CNN)

# Face detection params
MAX_CENTER_SHIFT_RATIO = 0.35
SMOOTH_ALPHA           = 0.4
WINDOW_STRIDE          = 5

# ── MEDIAPIPE ─────────────────────────────────────────────────────────────────
mp_face = mp.solutions.face_mesh

def make_detector():
    return mp_face.FaceMesh(
        static_image_mode=False,
        max_num_faces=3,
        refine_landmarks=False,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

# ── FACE DETECTION ────────────────────────────────────────────────────────────
def detect_faces(frame, detector):
    h, w = frame.shape[:2]
    rgb  = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res  = detector.process(rgb)
    if res.multi_face_landmarks is None:
        return []
    faces = []
    for lm in res.multi_face_landmarks:
        xs = [p.x for p in lm.landmark]
        ys = [p.y for p in lm.landmark]
        x1 = int(min(xs)*w); x2 = int(max(xs)*w)
        y1 = int(min(ys)*h); y2 = int(max(ys)*h)
        cx = (x1+x2)//2;    cy = (y1+y2)//2
        faces.append((x1,y1,x2,y2,cx,cy,lm))
    return faces


def pick_face(faces, prev_center=None, prev_size=None):
    """Pick best face — largest if no history, closest if tracking."""
    if not faces:
        return None
    if prev_center is None:
        areas = [(f[2]-f[0])*(f[3]-f[1]) for f in faces]
        return faces[int(np.argmax(areas))]
    dists = [np.hypot(f[4]-prev_center[0], f[5]-prev_center[1]) for f in faces]
    return faces[int(np.argmin(dists))]


# ── ALIGNED FACE CROP (for CNN) ───────────────────────────────────────────────
def align_and_crop(frame, lm):
    """Rotation-correct then landmark crop. Returns 224x224 BGR or None."""
    h, w = frame.shape[:2]
    left_eye  = [33,133];  right_eye = [362,263]
    lx = np.mean([lm.landmark[i].x for i in left_eye])*w
    ly = np.mean([lm.landmark[i].y for i in left_eye])*h
    rx = np.mean([lm.landmark[i].x for i in right_eye])*w
    ry = np.mean([lm.landmark[i].y for i in right_eye])*h
    angle  = np.degrees(np.arctan2(ry-ly, rx-lx))
    center = (int((lx+rx)/2), int((ly+ry)/2))
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(frame, M, (w,h))
    # Re-detect landmarks on aligned frame
    lx2 = np.mean([lm.landmark[i].x for i in left_eye])*w
    ly2 = np.mean([lm.landmark[i].y for i in left_eye])*h
    rx2 = np.mean([lm.landmark[i].x for i in right_eye])*w
    ry2 = np.mean([lm.landmark[i].y for i in right_eye])*h
    cx = int((lx2+rx2)/2); cy = int((ly2+ry2)/2)
    eye_dist = np.hypot(lx2-rx2, ly2-ry2)
    size = int(eye_dist*4.5); half = size//2
    x1=max(0,cx-half); y1=max(0,cy-half)
    x2=min(w,cx+half); y2=min(h,cy+half)
    crop = aligned[y1:y2, x1:x2]
    if crop.size == 0: return None
    return cv2.resize(crop, (OUTPUT_SIZE,OUTPUT_SIZE))


# ── RAW FACE CROP (for PRNU) — NO ALIGNMENT ──────────────────────────────────
def raw_crop(frame, face_box):
    """
    Minimal crop — face bounding box with 20% margin, no rotation.
    Preserves camera noise fingerprint (PRNU).
    Returns (crop_224x224, mean_rgb_before_resize).
    """
    h, w = frame.shape[:2]
    x1,y1,x2,y2 = face_box
    bw = x2-x1; bh = y2-y1
    margin_x = int(bw*0.2); margin_y = int(bh*0.2)
    rx1 = max(0, x1-margin_x); ry1 = max(0, y1-margin_y)
    rx2 = min(w, x2+margin_x); ry2 = min(h, y2+margin_y)
    crop = frame[ry1:ry2, rx1:rx2]
    if crop.size == 0:
        return None, None
    # Mean RGB BEFORE resize — this is the raw rPPG signal
    mean_rgb = np.mean(crop.reshape(-1,3), axis=0).astype(np.float32)  # [B,G,R]
    mean_rgb = mean_rgb[[2,1,0]]  # convert to [R,G,B]
    resized  = cv2.resize(crop, (OUTPUT_SIZE,OUTPUT_SIZE),
                          interpolation=cv2.INTER_AREA)  # AREA for downscale quality
    return resized, mean_rgb


# ── OPTICAL FLOW MOTION SCORE ─────────────────────────────────────────────────
def motion_score(frames):
    prev = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY)
    mags = []
    for f in frames[1:]:
        curr = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(
            prev, curr, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        mags.append(np.mean(np.sqrt(flow[...,0]**2 + flow[...,1]**2)))
        prev = curr
    return float(np.mean(mags)) if mags else 0.0


# ── SELECT BEST ALIGNED WINDOW ────────────────────────────────────────────────
def select_aligned_window(face_frames):
    """
    Sliding window over accumulated aligned face frames.
    Pick window with motion score closest to TARGET_MOTION.
    Same logic as original preprocessing.
    """
    best_seq  = None
    best_diff = float('inf')
    needed    = N_ALIGNED * TEMPORAL_GAP

    for start in range(0, len(face_frames)-needed+1, WINDOW_STRIDE):
        seg = [face_frames[start + i*TEMPORAL_GAP] for i in range(N_ALIGNED)
               if start + i*TEMPORAL_GAP < len(face_frames)]
        if len(seg) < N_ALIGNED: continue
        diff = abs(motion_score(seg) - TARGET_MOTION)
        if diff < best_diff:
            best_diff = diff; best_seq = seg

    return best_seq  # list of N_ALIGNED aligned BGR frames


# ── SELECT RAW WINDOW ─────────────────────────────────────────────────────────
def select_raw_window(raw_frames_rppg):
    """
    raw_frames_rppg: list of (raw_224_bgr, mean_rgb_float32) tuples
    Pick 64 consecutive entries from the middle of the clip
    (stable region with good face detection).
    """
    needed = N_RAW * RAW_GAP
    if len(raw_frames_rppg) < needed:
        # Not enough frames — use all available, evenly spaced
        indices = np.linspace(0, len(raw_frames_rppg)-1, N_RAW, dtype=int)
        return [raw_frames_rppg[i] for i in indices]

    # Take from middle for most stable face region
    mid   = len(raw_frames_rppg) // 2
    start = max(0, mid - needed//2)
    seq   = [raw_frames_rppg[start + i*RAW_GAP] for i in range(N_RAW)
             if start + i*RAW_GAP < len(raw_frames_rppg)]

    if len(seq) < N_RAW:
        # Fallback: pad from end
        while len(seq) < N_RAW:
            seq.append(raw_frames_rppg[-1])

    return seq[:N_RAW]


# ── PROCESS ONE VIDEO ─────────────────────────────────────────────────────────
def process_video(video_path, out_dir, detector):
    """
    Returns True on success, False on failure.
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot open"

    fps         = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Read all frames
    orig_frames = []
    while True:
        ret, frame = cap.read()
        if not ret: break
        orig_frames.append(frame)
    cap.release()

    if len(orig_frames) < 40:
        return False, f"too short ({len(orig_frames)} frames)"

    # ── Track face across all frames ──────────────────────────────────────────
    prev_center = None; prev_size = None
    aligned_face_frames  = []   # list of aligned 224x224 BGR
    raw_face_rppg        = []   # list of (raw_224_bgr, mean_rgb)
    used_frame_indices_a = []   # original frame index for each aligned frame
    used_frame_indices_r = []   # original frame index for each raw frame

    for fi, frame in enumerate(orig_frames):
        faces = detect_faces(frame, detector)
        face  = pick_face(faces, prev_center, prev_size)
        if face is None:
            continue

        x1,y1,x2,y2,cx,cy,lm = face
        size = max(x2-x1, y2-y1)

        # Smooth tracking
        if prev_center is not None:
            max_shift = prev_size * MAX_CENTER_SHIFT_RATIO
            shift     = np.hypot(cx-prev_center[0], cy-prev_center[1])
            if shift > max_shift:
                ratio = max_shift/shift
                cx = int(prev_center[0]+(cx-prev_center[0])*ratio)
                cy = int(prev_center[1]+(cy-prev_center[1])*ratio)
            size = int(SMOOTH_ALPHA*size + (1-SMOOTH_ALPHA)*prev_size)

        prev_center = (cx,cy); prev_size = size

        # Aligned crop (for CNN)
        aligned = align_and_crop(frame, lm)
        if aligned is not None:
            aligned_face_frames.append(aligned)
            used_frame_indices_a.append(fi)

        # Raw crop (for PRNU + rPPG) — every frame, no alignment
        rc, mean_rgb = raw_crop(frame, (x1,y1,x2,y2))
        if rc is not None:
            raw_face_rppg.append((rc, mean_rgb))
            used_frame_indices_r.append(fi)

    # ── Select aligned window ─────────────────────────────────────────────────
    if len(aligned_face_frames) < N_ALIGNED:
        return False, f"insufficient aligned frames ({len(aligned_face_frames)})"

    best_aligned = select_aligned_window(aligned_face_frames)
    if best_aligned is None:
        return False, "window selection failed"

    # ── Select raw window ─────────────────────────────────────────────────────
    if len(raw_face_rppg) < N_RAW:
        return False, f"insufficient raw frames ({len(raw_face_rppg)})"

    best_raw = select_raw_window(raw_face_rppg)

    # ── Save aligned frames ───────────────────────────────────────────────────
    aligned_dir = out_dir / "frames_aligned"
    aligned_dir.mkdir(parents=True, exist_ok=True)
    for i, frame in enumerate(best_aligned):
        cv2.imwrite(str(aligned_dir / f"{i:03d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

    # ── Save raw PNG frames ───────────────────────────────────────────────────
    raw_dir = out_dir / "frames_raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    rppg_signal = []
    for i, (rc, mean_rgb) in enumerate(best_raw):
        cv2.imwrite(str(raw_dir / f"{i:03d}.png"), rc)
        rppg_signal.append(mean_rgb)  # [R,G,B] float32

    # ── Save rPPG signal ──────────────────────────────────────────────────────
    rppg_arr = np.array(rppg_signal, dtype=np.float32)  # (N_RAW, 3)
    np.save(str(out_dir / "rppg.npy"), rppg_arr)

    # ── Save metadata ─────────────────────────────────────────────────────────
    meta = {
        'source_video': video_path.name,
        'fps': float(fps),
        'total_original_frames': total_frames,
        'effective_fps_rppg': float(fps / RAW_GAP),
        'n_aligned': N_ALIGNED,
        'n_raw': N_RAW,
        'temporal_gap_aligned': TEMPORAL_GAP,
        'temporal_gap_raw': RAW_GAP,
        'duration_rppg_seconds': float(N_RAW * RAW_GAP / fps),
    }
    with open(out_dir / "meta.json", 'w') as f:
        json.dump(meta, f, indent=2)

    return True, "ok"


# ── PROCESS FOLDER ────────────────────────────────────────────────────────────
def process_folder(video_dir, out_dir, label, type_name,
                   test_mode=False, resume=True):
    videos = sorted(video_dir.glob("*.mp4"))
    if not videos:
        print(f"  WARNING: No .mp4 files in {video_dir}")
        return 0, 0

    if test_mode:
        videos = videos[:5]

    ok_count   = 0
    fail_count = 0
    detector   = make_detector()  # fresh detector per folder

    for v in tqdm(videos, desc=f"  {type_name}", ncols=72, unit="vid"):
        # Clip name: type_stem, e.g. real_000 or Deepfakes_000_003
        stem      = v.stem
        clip_name = f"{type_name}_{stem}" if type_name != "real" else f"real_{stem}"
        clip_dir  = out_dir / label / clip_name

        # Resume: skip if already done (has frames_aligned with 20 frames)
        if resume and clip_dir.exists():
            existing = list((clip_dir/"frames_aligned").glob("*.jpg"))
            if len(existing) >= N_ALIGNED:
                ok_count += 1
                continue

        success, reason = process_video(v, clip_dir, detector)
        if success:
            ok_count += 1
        else:
            fail_count += 1
            # Clean up partial output
            if clip_dir.exists():
                import shutil
                shutil.rmtree(clip_dir, ignore_errors=True)

    return ok_count, fail_count


# ── MAIN ──────────────────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true',
                        help='Process only 5 videos per type')
    parser.add_argument('--resume', action='store_true', default=True,
                        help='Skip already-processed clips (default: True)')
    parser.add_argument('--no_resume', action='store_true',
                        help='Reprocess everything from scratch')
    args = parser.parse_args()
    resume = not args.no_resume

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  FF++ C23 — Final Preprocessing")
    print("=" * 60)
    print(f"  Source      : {DATA_ROOT}")
    print(f"  Output      : {OUTPUT_ROOT}")
    print(f"  Test mode   : {args.test_mode}")
    print(f"  Resume      : {resume}")
    print(f"  Aligned     : {N_ALIGNED} frames (gap={TEMPORAL_GAP}) → JPG q{JPEG_QUALITY}")
    print(f"  Raw PNG     : {N_RAW} frames (gap={RAW_GAP}) → PNG lossless")
    print(f"  rPPG signal : {N_RAW} frames → .npy (R,G,B means)")
    print(f"  Effective fps for rPPG: ~12.5 Hz (5.12s window)")
    print()

    total_ok = 0; total_fail = 0

    # Real videos
    print("Processing REAL videos...")
    ok, fail = process_folder(
        DATA_ROOT / "real", OUTPUT_ROOT, "real", "real",
        test_mode=args.test_mode, resume=resume)
    print(f"  Real: {ok} ok, {fail} failed")
    total_ok += ok; total_fail += fail

    # Fake videos
    for ft in FAKE_TYPES:
        print(f"\nProcessing FAKE — {ft}...")
        ok, fail = process_folder(
            DATA_ROOT / "fake" / ft, OUTPUT_ROOT, "fake", ft,
            test_mode=args.test_mode, resume=resume)
        print(f"  {ft}: {ok} ok, {fail} failed")
        total_ok += ok; total_fail += fail

    print(f"\n{'='*60}")
    print(f"  COMPLETE — {total_ok} clips processed, {total_fail} failed")
    print(f"  Output: {OUTPUT_ROOT}")
    print(f"\n  Structure per clip:")
    print(f"    frames_aligned/  — {N_ALIGNED} aligned JPGs for CNN")
    print(f"    frames_raw/      — {N_RAW} unaligned PNGs for PRNU")
    print(f"    rppg.npy         — ({N_RAW},3) float32 R,G,B means for rPPG")
    print(f"    meta.json        — fps, frame info")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()