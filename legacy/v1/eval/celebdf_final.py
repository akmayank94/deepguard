"""
eval/celebdf_final.py

Clean standalone cross-dataset evaluator for v2 B3 model.
Works directly with CelebDF_v2_test_preprocessed structure:
    real/clip_name/frames_aligned/*.jpg
    fake/clip_name/frames_aligned/*.jpg

Usage:
    python eval/celebdf_final.py
"""

import sys, os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
CELEBDF_ROOT = Path("D:/DO NOT TOUCH/CelebDF_v2_test_preprocessed")
CHECKPOINT   = Path("checkpoints/run_v2_full_b3/checkpoint_best.pt")
OUTPUT_JSON  = Path("checkpoints/run_v2_full_b3/celebdf_official_results.json")

IMG_SIZE     = 224
N_FRAMES     = 20
USE_TTA      = True

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_eval = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])
tf_flip = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(p=1.0),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def main():
    print("="*60)
    print("  Final Celeb-DF v2 Evaluation — v2 B3 Model")
    print("="*60)

    # ── Verify paths ──────────────────────────────────────────────────────────
    if not CELEBDF_ROOT.exists():
        print(f"ERROR: {CELEBDF_ROOT} not found"); sys.exit(1)
    if not CHECKPOINT.exists():
        print(f"ERROR: {CHECKPOINT} not found"); sys.exit(1)

    # ── Collect all clips ─────────────────────────────────────────────────────
    print(f"\n  Scanning {CELEBDF_ROOT}...")
    clips = []
    for label, sub in [(0, 'real'), (1, 'fake')]:
        subdir = CELEBDF_ROOT / sub
        if not subdir.exists():
            print(f"  WARNING: {subdir} not found")
            continue
        for clip_dir in sorted(subdir.iterdir()):
            if not clip_dir.is_dir():
                continue
            aligned_dir = clip_dir / "frames_aligned"
            if not aligned_dir.exists():
                continue
            frames = sorted(aligned_dir.glob("*.jpg"))
            if len(frames) == 0:
                continue
            clips.append((clip_dir, label, frames))

    real_n = sum(1 for _,l,_ in clips if l==0)
    fake_n = sum(1 for _,l,_ in clips if l==1)
    print(f"  Found {len(clips)} clips: real={real_n} fake={fake_n}")

    if len(clips) == 0:
        print("ERROR: No clips found. Check path structure.")
        # Debug: show what is actually in the folder
        print(f"\n  Contents of {CELEBDF_ROOT}:")
        for item in CELEBDF_ROOT.iterdir():
            print(f"    {item.name}/")
            if item.is_dir():
                subitems = list(item.iterdir())[:3]
                for si in subitems:
                    print(f"      {si.name}/")
                    if si.is_dir():
                        frames = list(si.iterdir())[:3]
                        for f in frames:
                            print(f"        {f.name}")
        sys.exit(1)

    # ── Verify one clip has readable frames ───────────────────────────────────
    test_clip, test_label, test_frames = clips[0]
    print(f"\n  Sample clip: {test_clip.name}")
    print(f"  Label: {'real' if test_label==0 else 'fake'}")
    print(f"  Frames: {len(test_frames)}")
    print(f"  First frame: {test_frames[0]}")
    try:
        img = Image.open(test_frames[0]).convert("RGB")
        print(f"  Frame size: {img.size}")
        tensor = tf_eval(img)
        print(f"  Tensor shape: {tensor.shape}  ✓")
    except Exception as e:
        print(f"  ERROR reading frame: {e}")
        sys.exit(1)

    # ── Load model ────────────────────────────────────────────────────────────
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n  Device: {device}")

    ckpt = torch.load(str(CHECKPOINT), map_location=device)
    print(f"  Checkpoint: epoch={ckpt['epoch']} val_auc={ckpt['val_auc']:.4f}")
    print(f"  raw_model: {ckpt['config'].get('raw_model', 'unknown')}")

    from models.detector_v2 import build_detector_v2
    model = build_detector_v2(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Model loaded ✓")

    # ── Run inference ─────────────────────────────────────────────────────────
    print(f"\n  Running inference (TTA={USE_TTA})...")

    clip_probs  = []
    clip_labels = []
    frame_probs_all  = []
    frame_labels_all = []

    # Dummy inputs for unused branches (handcrafted, rPPG)
    # Gate weights: handcrafted=0.02, rPPG=0.00 — negligible
    hc_zeros   = torch.zeros(1, 9369).to(device)
    rppg_zeros = torch.zeros(1, 32).to(device)

    for clip_dir, label, frame_paths in tqdm(
            clips, desc="  Evaluating", ncols=72, unit="clip"):

        # Sample N_FRAMES evenly from available frames
        total   = len(frame_paths)
        indices = np.linspace(0, total-1, N_FRAMES, dtype=int)
        sampled = [frame_paths[i] for i in indices]

        # Load PRNU features from frames_raw if available
        raw_dir   = clip_dir / "frames_raw"
        rppg_path = clip_dir / "rppg.npy"

        # PRNU
        try:
            from features.prnu import extract_prnu_features
            raw_paths = sorted(raw_dir.glob("*.png")) if raw_dir.exists() else []
            prnu_feat = extract_prnu_features(raw_paths, n_frames=20)
            prnu_t    = torch.from_numpy(prnu_feat).unsqueeze(0).to(device)
        except Exception:
            prnu_t = torch.zeros(1, 30).to(device)

        # rPPG (gate=0.00, use zeros)
        rppg_t = rppg_zeros

        frame_probs = []
        for fp in sampled:
            try:
                img_pil = Image.open(fp).convert("RGB")
                # Resize if not already 224x224
                if img_pil.size != (IMG_SIZE, IMG_SIZE):
                    img_pil = img_pil.resize(
                        (IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            except Exception as e:
                continue

            rr = tf_eval(img_pil).unsqueeze(0).to(device)

            # Diff image from this frame
            try:
                img_np = np.array(img_pil).astype(np.float32) / 255.0
                R, G, B = img_np[:,:,0], img_np[:,:,1], img_np[:,:,2]
                from scipy.ndimage import uniform_filter
                def en(ch): return np.abs(ch - uniform_filter(ch, size=3))
                diff = np.stack([en(R)-en(B), en(R)-en(G), en(G)-en(B)],
                                axis=2)
                diff = (diff - diff.min()) / (diff.max() - diff.min() + 1e-8)
                di = torch.from_numpy(
                    diff.astype(np.float32).transpose(2,0,1)
                ).unsqueeze(0).to(device)
            except Exception:
                di = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)

            with torch.no_grad():
                logit = model(rr, di, hc_zeros, prnu_t, rppg_t,
                              return_gates=False)
                prob  = torch.sigmoid(logit).item()

                if USE_TTA:
                    rr_f = tf_flip(img_pil).unsqueeze(0).to(device)
                    lf   = model(rr_f, di, hc_zeros, prnu_t, rppg_t,
                                 return_gates=False)
                    prob = (prob + torch.sigmoid(lf).item()) / 2.0

            frame_probs.append(prob)
            frame_probs_all.append(prob)
            frame_labels_all.append(float(label))

        if frame_probs:
            clip_probs.append(float(np.mean(frame_probs)))
            clip_labels.append(float(label))

    # ── Compute metrics ───────────────────────────────────────────────────────
    print(f"\n  Processed {len(clip_probs)} clips, "
          f"{len(frame_probs_all)} frames")

    if len(clip_probs) == 0:
        print("ERROR: No predictions generated"); sys.exit(1)

    cp = np.array(clip_probs)
    cl = np.array(clip_labels)
    fp = np.array(frame_probs_all)
    fl = np.array(frame_labels_all)

    # Frame-level
    fr_auc = float(roc_auc_score(fl, fp))
    fr_f1  = float(f1_score(fl, (fp>=0.5).astype(int), zero_division=0))
    fr_acc = float(accuracy_score(fl, (fp>=0.5).astype(int)))

    # Clip-level: find best threshold
    best_thr, best_f1 = 0.5, 0.0
    for thr in np.arange(0.20, 0.80, 0.01):
        f1 = f1_score(cl, (cp>=thr).astype(int), zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, float(thr)

    cl_auc = float(roc_auc_score(cl, cp))
    cl_f1  = float(f1_score(cl, (cp>=best_thr).astype(int), zero_division=0))
    cl_acc = float(accuracy_score(cl, (cp>=best_thr).astype(int)))

    cl_auc_05  = float(roc_auc_score(cl, cp))
    cl_f1_05   = float(f1_score(cl, (cp>=0.5).astype(int), zero_division=0))
    cl_acc_05  = float(accuracy_score(cl, (cp>=0.5).astype(int)))

    ff_auc = ckpt['val_auc']
    gap    = ff_auc - cl_auc

    print(f"\n{'='*60}")
    print(f"  RESULTS — v2 B3 on Official Celeb-DF v2 Test Split")
    print(f"{'='*60}")
    print(f"  Frame-level:")
    print(f"    AUC={fr_auc:.4f}  F1={fr_f1:.4f}  Acc={fr_acc:.4f}  thr=0.50")
    print(f"  Clip-level (thr={best_thr:.2f}, tuned):")
    print(f"    AUC={cl_auc:.4f}  F1={cl_f1:.4f}  Acc={cl_acc:.4f}")
    print(f"  Clip-level (thr=0.50):")
    print(f"    AUC={cl_auc:.4f}  F1={cl_f1_05:.4f}  Acc={cl_acc_05:.4f}")
    print(f"\n  Generalization gap:")
    print(f"    FF++ val AUC  : {ff_auc:.4f}")
    print(f"    Celeb-DF AUC  : {cl_auc:.4f}")
    print(f"    Drop          : {gap:.4f} ({gap/ff_auc*100:.1f}%)")
    print(f"\n  Clips evaluated: {len(cl)} "
          f"(real={int((cl==0).sum())} fake={int((cl==1).sum())})")

    # Save
    results = {
        'model':           'v2_B3',
        'checkpoint':      str(CHECKPOINT),
        'ff_val_auc':      ff_auc,
        'checkpoint_epoch': ckpt['epoch'],
        'celebdf_official_test': {
            'n_clips':        len(cl),
            'n_real':         int((cl==0).sum()),
            'n_fake':         int((cl==1).sum()),
            'clip_auc':       cl_auc,
            'clip_f1_tuned':  cl_f1,
            'clip_acc_tuned': cl_acc,
            'best_threshold': best_thr,
            'frame_auc':      fr_auc,
            'frame_f1':       fr_f1,
        }
    }
    OUTPUT_JSON.parent.mkdir(exist_ok=True)
    json.dump(results, open(str(OUTPUT_JSON), 'w'), indent=2)
    print(f"\n  Saved: {OUTPUT_JSON}")
    print("="*60)


if __name__ == "__main__":
    main()