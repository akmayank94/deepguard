"""
quick_fix_test.py

Tests whether clipping the handcrafted features at inference time
fixes the BatchNorm variance problem without recomputing the cache.

Also checks what the actual feature value distribution looks like
in the full cache to determine the right clip threshold.
"""
import h5py
import numpy as np
import os
from dotenv import load_dotenv
load_dotenv(override=True)

cache_dir  = os.environ.get("CACHE_DIR", "cache")
cache_path = f"{cache_dir}/features_full_seed42.h5"

print("Analyzing full cache feature distributions...")
print("(sampling 500 frames from train split)\n")

with h5py.File(cache_path, 'r') as f:
    labels = f['train']['labels'][:]
    np.random.seed(42)
    idx = np.random.choice(len(labels), 500, replace=False)
    idx_sorted = sorted(idx)

    hc = f['train']['handcrafted'][idx_sorted].astype(np.float32)
    en = f['train']['energy'][idx_sorted].astype(np.float32)
    dc = f['train']['dccm'][idx_sorted].astype(np.float32)
    lb = labels[idx_sorted]

real_hc = hc[lb == 0]
fake_hc = hc[lb == 1]

print("=== Handcrafted feature stats ===")
print(f"  Overall — min:{hc.min():.1f}  max:{hc.max():.1f}  mean:{hc.mean():.4f}  std:{hc.std():.4f}")
print(f"  Real    — mean:{real_hc.mean():.4f}  std:{real_hc.std():.4f}")
print(f"  Fake    — mean:{fake_hc.mean():.4f}  std:{fake_hc.std():.4f}")

# Check percentiles to find good clip threshold
percs = [50, 90, 95, 99, 99.5, 99.9, 100]
abs_vals = np.abs(hc).flatten()
print(f"\n  Absolute value percentiles:")
for p in percs:
    print(f"    {p:5.1f}th percentile: {np.percentile(abs_vals, p):.2f}")

print("\n=== Energy feature stats ===")
en_abs = np.abs(en).flatten()
print(f"  Overall — min:{en.min():.1f}  max:{en.max():.1f}")
print(f"  99th pct: {np.percentile(en_abs, 99):.2f}")
print(f"  99.9th pct: {np.percentile(en_abs, 99.9):.2f}")

print("\n=== DCCM feature stats ===")
dc_abs = np.abs(dc).flatten()
print(f"  Overall — min:{dc.min():.4f}  max:{dc.max():.4f}")
print(f"  99th pct: {np.percentile(dc_abs, 99):.6f}")

# Check separability with clipping at different thresholds
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

print("\n=== LogReg AUC with different clip thresholds ===")
# Need val split
with h5py.File(cache_path, 'r') as f:
    val_labels = f['val']['labels'][:]
    np.random.seed(0)
    vidx = np.random.choice(len(val_labels), min(1000, len(val_labels)), replace=False)
    vidx_s = sorted(vidx)
    val_hc = f['val']['handcrafted'][vidx_s].astype(np.float32)
    val_lb = val_labels[vidx_s]

# Use full train sample for fitting
with h5py.File(cache_path, 'r') as f:
    tr_labels = f['train']['labels'][:]
    tidx = np.random.choice(len(tr_labels), min(2000, len(tr_labels)), replace=False)
    tidx_s = sorted(tidx)
    tr_hc = f['train']['handcrafted'][tidx_s].astype(np.float32)
    tr_lb = tr_labels[tidx_s]

for clip in [None, 1000, 500, 200, 100, 50]:
    tr_c = np.clip(tr_hc, -clip, clip) if clip else tr_hc
    va_c = np.clip(val_hc, -clip, clip) if clip else val_hc
    sc = StandardScaler()
    tr_s = sc.fit_transform(tr_c)
    va_s = sc.transform(va_c)
    lr = LogisticRegression(C=0.1, max_iter=300, random_state=42)
    lr.fit(tr_s, tr_lb)
    auc = roc_auc_score(val_lb, lr.predict_proba(va_s)[:, 1])
    clip_str = f"clip={clip:>6}" if clip else "  no clip"
    print(f"  {clip_str}: val AUC = {auc:.4f}")

print("\nRecommendation: use the clip value that gives highest val AUC.")
print("We will apply this clip in the CachedFeatureDataset so no cache recompute needed.")