"""
diag3_model_behavior.py

Run this AFTER at least 5 epochs of training to understand what the
model has actually learned.

Checks:
1. Per-manipulation-type predictions — does the model discriminate
   some forgery types better than others?
2. Confidence distribution — is the model outputting confident
   predictions or stuck near 0.5?
3. Gate values — which branch is the model relying on most?
4. Frame consistency — do frames from the same clip agree on prediction?
"""
import sys, os
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import torch
import h5py
from PIL import Image
import torchvision.transforms as T
from sklearn.metrics import roc_auc_score
from collections import defaultdict

cache_dir  = os.environ.get("CACHE_DIR", "cache")
data_root  = Path(os.environ.get("DATA_ROOT", ""))
model_dir  = os.environ.get("MODEL_DIR", "checkpoints")
cache_path = Path(cache_dir) / "features_full_seed42.h5"
ckpt_path  = Path(model_dir) / "run_full_hybrid" / "checkpoint_best.pt"

if not ckpt_path.exists():
    print(f"No checkpoint found at {ckpt_path}")
    print("Run after at least a few training epochs.")
    sys.exit(0)

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
tf_eval = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def load_frame(clip_name, label, frame_idx):
    sub  = "real" if label == 0 else "fake"
    path = data_root / sub / clip_name / f"{frame_idx:03d}.jpg"
    if not path.exists():
        jpgs = sorted((data_root/sub/clip_name).glob("*.jpg"))
        if not jpgs: return torch.zeros(3, 224, 224)
        path = jpgs[0]
    try:
        return tf_eval(Image.open(path).convert("RGB"))
    except:
        return torch.zeros(3, 224, 224)

print("="*60)
print("  Model Behavior Analysis")
print("="*60)

# Load model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
ckpt   = torch.load(ckpt_path, map_location=device)
print(f"\n  Checkpoint: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")

from models.detector import build_detector
model = build_detector(ckpt['config']).to(device)
model.load_state_dict(ckpt['model'])
model.eval()

# Run on 300 val samples
print(f"\n  Running inference on 300 val frames...")
all_probs, all_labels, all_types, all_clips = [], [], [], []
all_gates = []

with h5py.File(cache_path, 'r') as f:
    np.random.seed(0)
    va_labels = f['val']['labels'][:]
    ri = np.random.choice(np.where(va_labels==0)[0], 150, replace=False)
    fi = np.random.choice(np.where(va_labels==1)[0], 150, replace=False)
    idx = np.concatenate([ri, fi])

    clips  = [c.decode() if isinstance(c,bytes) else c
              for c in f['val']['clip_names'][idx]]
    labels = va_labels[idx]
    types  = [t.decode() if isinstance(t,bytes) else t
              for t in f['val']['manip_types'][idx]]
    frames = f['val']['frame_indices'][idx].astype(int)
    hc_raw = np.clip(f['val']['handcrafted'][idx], -500, 500).astype(np.float32)
    di_raw = f['val']['diff_images'][idx].astype(np.float32)

with torch.no_grad():
    for i in range(len(idx)):
        rr = load_frame(clips[i], labels[i], frames[i]).unsqueeze(0).to(device)
        di = torch.from_numpy(di_raw[i].transpose(2,0,1)).unsqueeze(0).to(device)
        hc = torch.from_numpy(hc_raw[i]).unsqueeze(0).to(device)
        logits, gates = model(rr, di, hc, return_gates=True)
        prob = torch.sigmoid(logits).item()
        all_probs.append(prob)
        all_gates.append(gates.cpu().numpy()[0])
        all_labels.append(labels[i])
        all_types.append(types[i])
        all_clips.append(clips[i])

all_probs  = np.array(all_probs)
all_labels = np.array(all_labels)
all_gates  = np.array(all_gates)   # (N, 3)

# 1. Overall AUC
overall_auc = roc_auc_score(all_labels, all_probs)
print(f"\n  Overall val AUC (300 samples): {overall_auc:.4f}")

# 2. Per-type AUC
print(f"\n  Per-manipulation AUC:")
for mt in sorted(set(all_types)):
    if mt == 'real': continue
    idx_mt = [i for i,t in enumerate(all_types) if t in (mt,'real')]
    if len(idx_mt) < 20: continue
    ml = all_labels[idx_mt]
    mp = all_probs[idx_mt]
    if len(np.unique(ml)) < 2: continue
    auc = roc_auc_score(ml, mp)
    print(f"    {mt:<22}: {auc:.4f}")

# 3. Confidence distribution
print(f"\n  Prediction confidence:")
print(f"    Real  probs — mean={all_probs[all_labels==0].mean():.3f}  "
      f"std={all_probs[all_labels==0].std():.3f}")
print(f"    Fake  probs — mean={all_probs[all_labels==1].mean():.3f}  "
      f"std={all_probs[all_labels==1].std():.3f}")
near_50 = ((all_probs > 0.4) & (all_probs < 0.6)).mean()
print(f"    Fraction near 0.5 (uncertain): {near_50:.1%}")
if near_50 > 0.6:
    print("    WARNING: Model is mostly uncertain — not learning clear boundaries")
else:
    print("    OK: Model shows confidence in predictions")

# 4. Gate analysis
print(f"\n  Average gate weights (raw_rgb, diff_img, handcrafted):")
g_mean = all_gates.mean(axis=0)
print(f"    raw_rgb={g_mean[0]:.3f}  diff_img={g_mean[1]:.3f}  "
      f"handcrafted={g_mean[2]:.3f}")
dominant = ['raw_rgb','diff_img','handcrafted'][np.argmax(g_mean)]
print(f"    Dominant branch: {dominant}")

# 5. Clip consistency
print(f"\n  Clip-level consistency:")
clip_preds = defaultdict(list)
for prob, label, clip in zip(all_probs, all_labels, all_clips):
    clip_preds[clip].append((prob, label))
consistent = 0
total_clips = 0
for clip, preds in clip_preds.items():
    if len(preds) > 1:
        total_clips += 1
        ps = [p for p,l in preds]
        consistent += (np.std(ps) < 0.15)
if total_clips > 0:
    print(f"    {consistent}/{total_clips} clips have consistent predictions (std<0.15)")

print(f"\n{'='*60}")
if overall_auc > 0.70:
    print("  GOOD: Model is learning real signal. Continue training.")
elif overall_auc > 0.55:
    print("  PARTIAL: Some signal. Val AUC may improve with more epochs.")
else:
    print("  POOR: Model not learning generalizable features.")
print("="*60)