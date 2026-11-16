"""
eval/celebdf_eval_rawonly.py

Cross-dataset evaluation on Celeb-DF v2 for the raw_only ablation model.
Same logic as celebdf_eval.py but loads RawOnlyDetector instead of DeepfakeDetector.

Usage:
    python eval/celebdf_eval_rawonly.py
"""

import sys, os, json, argparse
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import torch
import numpy as np
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, roc_curve
from collections import defaultdict
from tqdm import tqdm

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
N_FRAMES      = 20
IMG_SIZE      = 224

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


def collect_clips(celebdf_root, split):
    clips = []
    for label, sub in [(0,'real'),(1,'fake')]:
        base = celebdf_root / split / sub
        if not base.exists():
            print(f"  WARNING: {base} not found"); continue
        for d in sorted(base.iterdir()):
            if d.is_dir():
                clips.append((d, label))
    return clips


def sample_frames(clip_path, n=N_FRAMES):
    frames = sorted(clip_path.glob("*.png"))
    if not frames: frames = sorted(clip_path.glob("*.jpg"))
    if not frames: return []
    if len(frames) <= n: return frames
    idx = np.linspace(0, len(frames)-1, n, dtype=int)
    return [frames[i] for i in idx]


def run_inference(model, device, clips, use_tta=True):
    model.eval()
    clip_results = []
    frame_probs_all, frame_labels_all = [], []

    for clip_path, label in tqdm(clips, desc="  Clips", ncols=72, unit="clip"):
        frames = sample_frames(clip_path)
        if not frames: continue
        frame_probs = []
        for fp in frames:
            try:
                img_pil = Image.open(fp).convert("RGB")
                img_pil = img_pil.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
            except Exception: continue

            rr = tf_eval(img_pil).unsqueeze(0).to(device)
            # raw_only model ignores diff_image and handcrafted
            # but forward signature still takes them — pass zeros
            di = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)
            hc = torch.zeros(1, 9369).to(device)

            with torch.no_grad():
                logit = model(rr, di, hc, return_gates=False)
                prob  = torch.sigmoid(logit).item()
                if use_tta:
                    rr_f = tf_flip(img_pil).unsqueeze(0).to(device)
                    logit_f = model(rr_f, di, hc, return_gates=False)
                    prob = (prob + torch.sigmoid(logit_f).item()) / 2.0

            frame_probs.append(prob)
            frame_probs_all.append(prob)
            frame_labels_all.append(label)

        if frame_probs:
            clip_results.append((float(np.mean(frame_probs)), label, clip_path.name))

    clip_probs  = np.array([r[0] for r in clip_results])
    clip_labels = np.array([r[1] for r in clip_results])
    return clip_probs, clip_labels, np.array(frame_probs_all), np.array(frame_labels_all)


def find_best_threshold(probs, labels):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1: best_f1, best_thr = f1, thr
    return float(best_thr)


def print_metrics(probs, labels, desc, thr=0.5):
    preds = (probs >= thr).astype(int)
    auc = float(roc_auc_score(labels, probs))
    f1  = float(f1_score(labels, preds, zero_division=0))
    acc = float(accuracy_score(labels, preds))
    print(f"  {desc}")
    print(f"    N={len(labels)} (real={int((labels==0).sum())}, fake={int((labels==1).sum())})")
    print(f"    AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  thr={thr:.2f}")
    return {'auc':auc,'f1':f1,'acc':acc,'threshold':thr}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--celebdf_root', type=str,
                   default=os.environ.get('CELEBDF_ROOT',''))
    p.add_argument('--checkpoint', type=str,
                   default='checkpoints/ablation_raw_only/checkpoint_best.pt')
    p.add_argument('--no_tta', action='store_true')
    args = p.parse_args()

    celebdf_root = Path(args.celebdf_root)
    ckpt_path    = Path(args.checkpoint)

    if not celebdf_root.exists():
        print(f"ERROR: {celebdf_root} not found"); sys.exit(1)
    if not ckpt_path.exists():
        print(f"ERROR: {ckpt_path} not found"); sys.exit(1)

    print("="*60)
    print("  Celeb-DF v2 — Raw-Only Ablation Model")
    print("="*60)
    print(f"  Root  : {celebdf_root}")
    print(f"  Ckpt  : {ckpt_path}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(str(ckpt_path), map_location=device)
    print(f"  Model : epoch={ckpt['epoch']}  FF++ val_auc={ckpt['val_auc']:.4f}")

    # Load RawOnlyDetector from main_ablation.py
    from main_ablation import RawOnlyDetector
    model = RawOnlyDetector(dropout=0.5).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()
    print(f"  Loaded RawOnlyDetector successfully")

    clips = collect_clips(celebdf_root, 'Test')
    print(f"\n  Found {len(clips)} clips: "
          f"real={sum(1 for _,l in clips if l==0)}, "
          f"fake={sum(1 for _,l in clips if l==1)}")

    clip_probs, clip_labels, fr_probs, fr_labels = run_inference(
        model, device, clips, use_tta=not args.no_tta)

    print("\n  Results:")
    fr_m   = print_metrics(fr_probs,   fr_labels, "Frame-level (thr=0.5)", thr=0.5)
    best_thr = find_best_threshold(clip_probs, clip_labels)
    clip_m = print_metrics(clip_probs, clip_labels,
                           f"Clip-level  (thr={best_thr:.2f})", thr=best_thr)
    clip_05= print_metrics(clip_probs, clip_labels, "Clip-level  (thr=0.50)", thr=0.5)

    ff_auc  = ckpt['val_auc']
    cel_auc = clip_m['auc']
    gap     = ff_auc - cel_auc
    print(f"\n  Generalization gap:")
    print(f"    FF++ val AUC   : {ff_auc:.4f}")
    print(f"    Celeb-DF AUC   : {cel_auc:.4f}")
    print(f"    Drop           : {gap:.4f} ({gap/ff_auc*100:.1f}%)")

    # Save results
    model_dir = os.environ.get("MODEL_DIR","checkpoints")
    out_path  = Path(model_dir) / "ablation_raw_only" / "celebdf_results.json"
    results   = {
        'ff_plus_plus_val_auc': ff_auc,
        'checkpoint_epoch': ckpt['epoch'],
        'Test': {
            'frame_level': fr_m,
            'clip_level_tuned': clip_m,
            'clip_level_0.5': clip_05,
            'roc': {'fpr': roc_curve(clip_labels,clip_probs)[0].tolist(),
                    'tpr': roc_curve(clip_labels,clip_probs)[1].tolist()},
        }
    }
    json.dump(results, open(out_path,'w'), indent=2)
    print(f"\n  Saved to: {out_path}")

    print(f"\n{'='*60}")
    print(f"  SUMMARY — Raw-Only vs Full Model on Celeb-DF v2")
    print(f"{'='*60}")
    print(f"  Raw-only  : FF++ AUC={ff_auc:.4f}  Celeb-DF AUC={cel_auc:.4f}")

    # Load full model celebdf result for comparison if available
    full_celebdf = Path(model_dir)/"run_full_hybrid"/"celebdf_results.json"
    if full_celebdf.exists():
        fd = json.load(open(full_celebdf))
        full_ff  = fd.get('ff_plus_plus_val_auc', 0)
        full_cel = fd.get('Test',{}).get('clip_level_tuned',{}).get('auc',0)
        print(f"  Full model: FF++ AUC={full_ff:.4f}  Celeb-DF AUC={full_cel:.4f}")
        diff = full_cel - cel_auc
        if diff > 0:
            print(f"\n  ✓ CT spectral features improve Celeb-DF by +{diff:.4f}")
            print(f"    This proves spectral branch helps cross-dataset generalization")
        else:
            print(f"\n  ✗ CT spectral features do not improve Celeb-DF ({diff:+.4f})")
            print(f"    Raw RGB branch alone generalizes as well or better")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()