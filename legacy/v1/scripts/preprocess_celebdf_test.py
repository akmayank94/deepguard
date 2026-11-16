"""
scripts/preprocess_celebdf_test.py

Preprocesses ONLY the official test split of Celeb-DF v2.
Uses List_of_testing_videos.txt (518 videos) to identify test clips.

Uses IDENTICAL preprocessing to FF++C23_preprocessed:
  - MediaPipe face detection + alignment
  - frames_aligned/  20 aligned 224x224 JPGs (gap=2)
  - frames_raw/      64 unaligned 224x224 PNGs (gap=2)
  - rppg.npy         (64,3) float32 mean RGB
  - meta.json        fps, frame info

Output structure:
  CelebDF_v2_test_preprocessed/
    real/
      YouTube-real_00170/  frames_aligned/ frames_raw/ rppg.npy meta.json
      Celeb-real_id0_0000/ ...
    fake/
      Celeb-synthesis_id0_id16_0003/ ...

Label from list file:
  1 = real (YouTube-real and Celeb-real)
  0 = fake (Celeb-synthesis)

Usage:
    python scripts/preprocess_celebdf_test.py
    python scripts/preprocess_celebdf_test.py --test_mode  (first 10 videos)
"""

import cv2
import numpy as np
import json
import argparse
import shutil
from pathlib import Path
import mediapipe as mp
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CELEBDF_ROOT   = Path("D:/DO NOT TOUCH/CelebDF_v2")
LIST_FILE      = CELEBDF_ROOT / "List_of_testing_videos.txt"
OUTPUT_ROOT    = Path("D:/DO NOT TOUCH/CelebDF_v2_test_preprocessed")

# Same parameters as FF++C23_preprocessed
N_ALIGNED      = 20
TEMPORAL_GAP   = 2
TARGET_MOTION  = 0.8
N_RAW          = 64
RAW_GAP        = 2
OUTPUT_SIZE    = 224
JPEG_QUALITY   = 95
SMOOTH_ALPHA   = 0.4
MAX_CENTER_SHIFT_RATIO = 0.35
WINDOW_STRIDE  = 5


def parse_list_file(list_path):
    """
    Parse List_of_testing_videos.txt.
    Format: label path
      1 = real, 0 = fake
    Returns list of (video_path, label, clip_name)
    """
    entries = []
    with open(list_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            label    = int(parts[0])
            rel_path = parts[1]
            video_path = CELEBDF_ROOT / rel_path
            # Build clip name from folder and stem
            folder   = Path(rel_path).parent.name
            stem     = Path(rel_path).stem
            clip_name = f"{folder}_{stem}"
            entries.append((video_path, label, clip_name))
    return entries


def make_detector():
    return mp.solutions.face_mesh.FaceMesh(
        static_image_mode=False, max_num_faces=3,
        refine_landmarks=False,
        min_detection_confidence=0.5, min_tracking_confidence=0.5)


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
        x1,y1 = int(min(xs)*w), int(min(ys)*h)
        x2,y2 = int(max(xs)*w), int(max(ys)*h)
        cx,cy = (x1+x2)//2, (y1+y2)//2
        faces.append((x1,y1,x2,y2,cx,cy,lm))
    return faces


def pick_face(faces, prev_center=None, prev_size=None):
    if not faces: return None
    if prev_center is None:
        areas = [(f[2]-f[0])*(f[3]-f[1]) for f in faces]
        return faces[int(np.argmax(areas))]
    dists = [np.hypot(f[4]-prev_center[0],f[5]-prev_center[1]) for f in faces]
    return faces[int(np.argmin(dists))]


def align_and_crop(frame, lm):
    h,w = frame.shape[:2]
    L,R = [33,133],[362,263]
    lx = np.mean([lm.landmark[i].x for i in L])*w
    ly = np.mean([lm.landmark[i].y for i in L])*h
    rx = np.mean([lm.landmark[i].x for i in R])*w
    ry = np.mean([lm.landmark[i].y for i in R])*h
    angle  = np.degrees(np.arctan2(ry-ly,rx-lx))
    center = (int((lx+rx)/2), int((ly+ry)/2))
    M      = cv2.getRotationMatrix2D(center, angle, 1.0)
    aligned = cv2.warpAffine(frame, M, (w,h))
    eye_dist = np.hypot(lx-rx, ly-ry)
    size = int(eye_dist*4.5); half = size//2
    cx2,cy2 = center
    x1=max(0,cx2-half); y1=max(0,cy2-half)
    x2=min(w,cx2+half); y2=min(h,cy2+half)
    crop = aligned[y1:y2,x1:x2]
    if crop.size == 0: return None
    return cv2.resize(crop,(OUTPUT_SIZE,OUTPUT_SIZE))


def raw_crop(frame, face_box):
    h,w = frame.shape[:2]
    x1,y1,x2,y2 = face_box
    bw,bh = x2-x1,y2-y1
    mx,my = int(bw*0.2),int(bh*0.2)
    rx1=max(0,x1-mx); ry1=max(0,y1-my)
    rx2=min(w,x2+mx); ry2=min(h,y2+my)
    crop = frame[ry1:ry2,rx1:rx2]
    if crop.size==0: return None,None
    mean_rgb = crop.reshape(-1,3).mean(axis=0).astype(np.float32)[[2,1,0]]
    return cv2.resize(crop,(OUTPUT_SIZE,OUTPUT_SIZE),
                      interpolation=cv2.INTER_AREA), mean_rgb


def motion_score(frames):
    prev = cv2.cvtColor(frames[0],cv2.COLOR_BGR2GRAY)
    mags = []
    for f in frames[1:]:
        curr = cv2.cvtColor(f,cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(prev,curr,None,0.5,3,15,3,5,1.2,0)
        mags.append(np.mean(np.sqrt(flow[...,0]**2+flow[...,1]**2)))
        prev = curr
    return float(np.mean(mags)) if mags else 0.0


def select_aligned_window(face_frames):
    best_seq,best_diff = None,float('inf')
    needed = N_ALIGNED*TEMPORAL_GAP
    for start in range(0,len(face_frames)-needed+1,WINDOW_STRIDE):
        seg = [face_frames[start+i*TEMPORAL_GAP] for i in range(N_ALIGNED)
               if start+i*TEMPORAL_GAP < len(face_frames)]
        if len(seg)<N_ALIGNED: continue
        diff = abs(motion_score(seg)-TARGET_MOTION)
        if diff<best_diff: best_diff,best_seq = diff,seg
    return best_seq


def select_raw_window(raw_list):
    needed = N_RAW*RAW_GAP
    if len(raw_list) < needed:
        idx = np.linspace(0,len(raw_list)-1,N_RAW,dtype=int)
        return [raw_list[i] for i in idx]
    mid   = len(raw_list)//2
    start = max(0,mid-needed//2)
    seq   = [raw_list[start+i*RAW_GAP] for i in range(N_RAW)
             if start+i*RAW_GAP < len(raw_list)]
    while len(seq)<N_RAW: seq.append(raw_list[-1])
    return seq[:N_RAW]


def process_video(video_path, out_dir, detector):
    if not video_path.exists():
        return False, f"video not found: {video_path}"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False, "cannot open"

    fps   = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    orig  = []
    while True:
        ret,frame = cap.read()
        if not ret: break
        orig.append(frame)
    cap.release()

    if len(orig) < 40:
        return False, f"too short ({len(orig)} frames)"

    prev_center,prev_size = None,None
    aligned_frames,raw_rppg = [],[]

    for frame in orig:
        faces = detect_faces(frame, detector)
        face  = pick_face(faces, prev_center, prev_size)
        if face is None: continue
        x1,y1,x2,y2,cx,cy,lm = face
        size = max(x2-x1,y2-y1)
        if prev_center is not None:
            max_shift = prev_size*MAX_CENTER_SHIFT_RATIO
            shift = np.hypot(cx-prev_center[0],cy-prev_center[1])
            if shift>max_shift:
                ratio = max_shift/shift
                cx = int(prev_center[0]+(cx-prev_center[0])*ratio)
                cy = int(prev_center[1]+(cy-prev_center[1])*ratio)
            size = int(SMOOTH_ALPHA*size+(1-SMOOTH_ALPHA)*prev_size)
        prev_center,prev_size = (cx,cy),size
        al = align_and_crop(frame,lm)
        if al is not None: aligned_frames.append(al)
        rc,mean_rgb = raw_crop(frame,(x1,y1,x2,y2))
        if rc is not None: raw_rppg.append((rc,mean_rgb))

    if len(aligned_frames)<N_ALIGNED:
        return False,f"insufficient aligned ({len(aligned_frames)})"
    if len(raw_rppg)<N_RAW:
        return False,f"insufficient raw ({len(raw_rppg)})"

    best_aligned = select_aligned_window(aligned_frames)
    if best_aligned is None: return False,"window selection failed"
    best_raw = select_raw_window(raw_rppg)

    adir = out_dir/"frames_aligned"; adir.mkdir(parents=True,exist_ok=True)
    for i,f in enumerate(best_aligned):
        cv2.imwrite(str(adir/f"{i:03d}.jpg"),f,
                    [cv2.IMWRITE_JPEG_QUALITY,JPEG_QUALITY])

    rdir = out_dir/"frames_raw"; rdir.mkdir(parents=True,exist_ok=True)
    rppg_signal = []
    for i,(rc,mean_rgb) in enumerate(best_raw):
        cv2.imwrite(str(rdir/f"{i:03d}.png"),rc)
        rppg_signal.append(mean_rgb)

    np.save(str(out_dir/"rppg.npy"),
            np.array(rppg_signal,dtype=np.float32))
    json.dump({
        'source_video': str(video_path.name),
        'fps': float(fps),
        'total_original_frames': total,
        'effective_fps_rppg': float(fps/RAW_GAP),
        'n_aligned': N_ALIGNED, 'n_raw': N_RAW,
        'temporal_gap_aligned': TEMPORAL_GAP,
        'temporal_gap_raw': RAW_GAP,
        'duration_rppg_seconds': float(N_RAW*RAW_GAP/fps),
    }, open(out_dir/"meta.json",'w'), indent=2)

    return True,"ok"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--test_mode', action='store_true',
                        help='Process only first 10 videos')
    args = parser.parse_args()

    OUTPUT_ROOT.mkdir(parents=True,exist_ok=True)
    (OUTPUT_ROOT/"real").mkdir(exist_ok=True)
    (OUTPUT_ROOT/"fake").mkdir(exist_ok=True)

    # Parse official test list
    entries = parse_list_file(LIST_FILE)
    print(f"  Official test list: {len(entries)} videos")

    real_count = sum(1 for _,l,_ in entries if l==1)
    fake_count = sum(1 for _,l,_ in entries if l==0)
    print(f"  Real: {real_count}  Fake: {fake_count}")

    # Show label distribution
    from collections import Counter
    folders = Counter(str(Path(v).parent.name) for v,l,c in entries)
    for folder,count in sorted(folders.items()):
        print(f"    {folder}: {count} videos")

    if args.test_mode:
        entries = entries[:10]
        print(f"\n  TEST MODE: processing only {len(entries)} videos")

    print(f"\n  Output: {OUTPUT_ROOT}")
    print(f"  Same preprocessing as FF++C23_preprocessed\n")

    detector = make_detector()
    ok,fail  = 0,0
    fail_list = []

    for video_path, label, clip_name in tqdm(
            entries, desc="  CelebDF test", ncols=72, unit="vid"):

        label_str = "real" if label==1 else "fake"
        clip_dir  = OUTPUT_ROOT / label_str / clip_name

        # Resume
        if clip_dir.exists():
            existing = list((clip_dir/"frames_aligned").glob("*.jpg"))
            if len(existing)>=N_ALIGNED:
                ok+=1; continue

        success,reason = process_video(video_path, clip_dir, detector)
        if success:
            ok+=1
        else:
            fail+=1
            fail_list.append(f"{clip_name}: {reason}")
            shutil.rmtree(clip_dir,ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"  COMPLETE: {ok} ok, {fail} failed")
    print(f"  Output: {OUTPUT_ROOT}")
    if fail_list[:5]:
        print(f"  Sample failures: {fail_list[:5]}")

    # Verify output
    real_clips = list((OUTPUT_ROOT/"real").iterdir())
    fake_clips = list((OUTPUT_ROOT/"fake").iterdir())
    print(f"\n  Verified output:")
    print(f"    Real clips: {len(real_clips)}")
    print(f"    Fake clips: {len(fake_clips)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()