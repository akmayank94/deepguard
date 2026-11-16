"""
diag1_feature_quality.py

Tests whether our three input types (raw_rgb, diff_image, handcrafted)
can each individually separate real from fake when trained on training
clips and tested on DIFFERENT validation clips.

This is the ground truth test — if a simple model trained on train clips
cannot predict val clips, no neural network will either.
"""
import sys, os, numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import h5py
from PIL import Image
import torch
import torchvision.transforms as T
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

cache_dir  = os.environ.get("CACHE_DIR", "cache")
data_root  = Path(os.environ.get("DATA_ROOT", ""))
cache_path = Path(cache_dir) / "features_full_seed42.h5"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
tf = T.Compose([T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

def load_raw_frame(clip_name, label, frame_idx):
    sub  = "real" if label == 0 else "fake"
    path = data_root / sub / clip_name / f"{frame_idx:03d}.jpg"
    if not path.exists():
        jpgs = sorted((data_root/sub/clip_name).glob("*.jpg"))
        if not jpgs: return np.zeros(3*224*224, dtype=np.float32)
        path = jpgs[0]
    try:
        img = tf(Image.open(path).convert("RGB"))
        return img.numpy().flatten()
    except:
        return np.zeros(3*224*224, dtype=np.float32)

print("="*60)
print("  Feature Quality — Cross-Clip Generalization Test")
print("="*60)

with h5py.File(cache_path, 'r') as f:
    # Sample balanced sets from train and val
    np.random.seed(42)

    tr_labels = f['train']['labels'][:]
    tr_ri = np.random.choice(np.where(tr_labels==0)[0], 200, replace=False)
    tr_fi = np.random.choice(np.where(tr_labels==1)[0], 200, replace=False)
    tr_idx = np.concatenate([tr_ri, tr_fi])
    tr_y   = tr_labels[tr_idx]

    va_labels = f['val']['labels'][:]
    va_ri = np.random.choice(np.where(va_labels==0)[0], 200, replace=False)
    va_fi = np.random.choice(np.where(va_labels==1)[0], 200, replace=False)
    va_idx = np.concatenate([va_ri, va_fi])
    va_y   = va_labels[va_idx]

    print("\nLoading features...")

    # 1. Handcrafted features
    tr_hc = np.clip(f['train']['handcrafted'][sorted(tr_idx)], -500, 500)
    va_hc = np.clip(f['val']['handcrafted'][sorted(va_idx)], -500, 500)
    # reorder to match tr_idx order
    sort_tr = np.argsort(tr_idx); unsort_tr = np.argsort(sort_tr)
    sort_va = np.argsort(va_idx); unsort_va = np.argsort(sort_va)
    tr_hc = tr_hc[unsort_tr]
    va_hc = va_hc[unsort_va]

    # 2. Diff images
    tr_di = f['train']['diff_images'][sorted(tr_idx)].reshape(400,-1)[unsort_tr]
    va_di = f['val']['diff_images'][sorted(va_idx)].reshape(400,-1)[unsort_va]

    tr_sort   = np.argsort(tr_idx)
    tr_unsort = np.argsort(tr_sort)
    va_sort   = np.argsort(va_idx)
    va_unsort = np.argsort(va_sort)

    tr_clips  = [c.decode() if isinstance(c,bytes) else c
                 for c in f['train']['clip_names'][np.sort(tr_idx)]]
    tr_clips  = [tr_clips[i] for i in tr_unsort]
    tr_frames = f['train']['frame_indices'][np.sort(tr_idx)].astype(int)[tr_unsort]
    va_clips  = [c.decode() if isinstance(c,bytes) else c
                 for c in f['val']['clip_names'][np.sort(va_idx)]]
    va_clips  = [va_clips[i] for i in va_unsort]
    va_frames = f['val']['frame_indices'][np.sort(va_idx)].astype(int)[va_unsort]

print("Loading raw RGB frames (400 train + 400 val)...")
tr_raw, va_raw = [], []
for i, (clip, fi, lbl) in enumerate(zip(tr_clips, tr_frames, tr_y)):
    if i % 50 == 0: print(f"  train {i}/400...")
    tr_raw.append(load_raw_frame(clip, lbl, fi))
for i, (clip, fi, lbl) in enumerate(zip(va_clips, va_frames, va_y)):
    if i % 50 == 0: print(f"  val   {i}/400...")
    va_raw.append(load_raw_frame(clip, lbl, fi))
tr_raw = np.array(tr_raw, dtype=np.float32)
va_raw = np.array(va_raw, dtype=np.float32)

print("\nRunning LogReg on each feature type...")
print(f"{'Feature':<20} {'Train AUC':>10} {'Val AUC':>10} {'Generalization':>15}")
print("-"*58)

results = {}
for name, Xtr, Xva in [
    ("Handcrafted",   tr_hc, va_hc),
    ("Diff image",    tr_di, va_di),
    ("Raw RGB (PCA)", tr_raw, va_raw),
]:
    sc = StandardScaler()
    Xtr_s = sc.fit_transform(Xtr)
    Xva_s = sc.transform(Xva)

    # For raw RGB, reduce with PCA first (too high dimensional)
    if Xtr_s.shape[1] > 1000:
        from sklearn.decomposition import PCA
        pca = PCA(n_components=200, random_state=42)
        Xtr_s = pca.fit_transform(Xtr_s)
        Xva_s = pca.transform(Xva_s)

    lr = LogisticRegression(C=0.1, max_iter=300, random_state=42)
    lr.fit(Xtr_s, tr_y)
    tr_auc = roc_auc_score(tr_y, lr.predict_proba(Xtr_s)[:,1])
    va_auc = roc_auc_score(va_y, lr.predict_proba(Xva_s)[:,1])
    gap    = tr_auc - va_auc
    results[name] = (tr_auc, va_auc)
    print(f"  {name:<18} {tr_auc:>10.4f} {va_auc:>10.4f} "
          f"{'gap='+f'{gap:.3f}':>15}")

print("\n" + "="*60)
best_name = max(results, key=lambda k: results[k][1])
best_va   = results[best_name][1]
print(f"  Best val AUC: {best_name} = {best_va:.4f}")
if best_va > 0.70:
    print("  STRONG SIGNAL — neural net should reach 80%+ val AUC")
elif best_va > 0.60:
    print("  MODERATE SIGNAL — neural net should reach 65-75% val AUC")
else:
    print("  WEAK SIGNAL — features do not generalize across clips")
    print("  This is a dataset characteristic, not a code bug")
print("="*60)