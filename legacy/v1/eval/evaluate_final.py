"""
eval/evaluate_final.py

Complete final evaluation with:
1. Standard evaluation (baseline)
2. Test-Time Augmentation (TTA) — averages predictions over flipped versions
3. Optimal threshold tuning from validation set
4. Full results table for paper
5. Per-manipulation AUC breakdown
6. Saves all results to JSON for plotting

Usage:
    python eval/evaluate_final.py
"""

import sys, os, json
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import torch
import numpy as np
import h5py
from PIL import Image
import torchvision.transforms as T
from torch.utils.data import DataLoader
from sklearn.metrics import (roc_auc_score, f1_score, accuracy_score,
                              roc_curve, confusion_matrix)
from collections import defaultdict
from tqdm import tqdm

# ── paths ──────────────────────────────────────────────────────────────
data_root  = Path(os.environ.get("DATA_ROOT", ""))
cache_dir  = os.environ.get("CACHE_DIR", "cache")
model_dir  = os.environ.get("MODEL_DIR", "checkpoints")
cache_path = Path(cache_dir) / "features_full_seed42.h5"
ckpt_path  = Path(model_dir) / "run_full_hybrid" / "checkpoint_best.pt"

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

tf_eval = T.Compose([T.ToTensor(),
                     T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
tf_flip = T.Compose([T.RandomHorizontalFlip(p=1.0),
                     T.ToTensor(),
                     T.Normalize(IMAGENET_MEAN, IMAGENET_STD)])

CLIP_LO, CLIP_HI = -500.0, 500.0

# ── helpers ────────────────────────────────────────────────────────────

def load_frame(clip_name, label, frame_idx, tf):
    sub  = "real" if label == 0 else "fake"
    path = data_root / sub / clip_name / f"{frame_idx:03d}.jpg"
    if not path.exists():
        jpgs = sorted((data_root / sub / clip_name).glob("*.jpg"))
        if not jpgs: return torch.zeros(3, 224, 224)
        path = jpgs[0]
    try:   return tf(Image.open(path).convert("RGB"))
    except: return torch.zeros(3, 224, 224)


def run_inference(model, device, cache_path, split,
                  use_tta=False, batch_size=32):
    """
    Run inference on a cache split.
    Returns (probs, labels, clip_names, manip_types).
    If use_tta=True, averages original + horizontally flipped prediction.
    """
    model.eval()
    all_probs, all_labels, all_clips, all_types = [], [], [], []

    with h5py.File(str(cache_path), 'r') as f:
        n          = f[split]['labels'].shape[0]
        labels     = f[split]['labels'][:].astype(int)
        clip_names = [c.decode() if isinstance(c,bytes) else c
                      for c in f[split]['clip_names'][:]]
        manip_types= [t.decode() if isinstance(t,bytes) else t
                      for t in f[split]['manip_types'][:]]
        frame_idxs = f[split]['frame_indices'][:].astype(int)
        hc_all     = np.clip(f[split]['handcrafted'][:],
                             CLIP_LO, CLIP_HI).astype(np.float32)
        di_all     = f[split]['diff_images'][:].astype(np.float32)

    with torch.no_grad():
        for i in tqdm(range(n), desc=f"  {split}", ncols=70, unit="fr"):
            clip, label, fi = clip_names[i], labels[i], frame_idxs[i]

            rr = load_frame(clip, label, fi, tf_eval).unsqueeze(0).to(device)
            di = torch.from_numpy(
                di_all[i].transpose(2,0,1)).unsqueeze(0).to(device)
            hc = torch.from_numpy(hc_all[i]).unsqueeze(0).to(device)

            logit = model(rr, di, hc, return_gates=False)
            prob  = torch.sigmoid(logit).item()

            if use_tta:
                rr_f = load_frame(clip, label, fi, tf_flip).unsqueeze(0).to(device)
                logit_f = model(rr_f, di, hc, return_gates=False)
                prob_f  = torch.sigmoid(logit_f).item()
                prob    = (prob + prob_f) / 2.0

            all_probs.append(prob)
            all_labels.append(label)
            all_clips.append(clip)
            all_types.append(manip_types[i])

    return (np.array(all_probs), np.array(all_labels),
            all_clips, all_types)


def compute_metrics(probs, labels, types, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    overall = {
        'auc':  float(roc_auc_score(labels, probs)),
        'f1':   float(f1_score(labels, preds, zero_division=0)),
        'acc':  float(accuracy_score(labels, preds)),
        'threshold': float(threshold),
        'n':    int(len(labels)),
    }
    per_type = {}
    for mt in sorted(set(types)):
        if mt == 'real': continue
        idx = [i for i,t in enumerate(types) if t in (mt,'real')]
        if len(idx) < 10: continue
        ml, mp = labels[idx], probs[idx]
        if len(np.unique(ml)) < 2: continue
        mp2 = (mp >= threshold).astype(int)
        try:
            per_type[mt] = {
                'auc': float(roc_auc_score(ml, mp)),
                'f1':  float(f1_score(ml, mp2, zero_division=0)),
                'acc': float(accuracy_score(ml, mp2)),
            }
        except: pass
    return overall, per_type


def find_best_threshold(val_probs, val_labels):
    """Find threshold that maximises F1 on validation set."""
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.3, 0.75, 0.01):
        preds = (val_probs >= thr).astype(int)
        f1 = f1_score(val_labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


# ── main ───────────────────────────────────────────────────────────────

def main():
    print("="*60)
    print("  Final Evaluation — Hybrid Deepfake Detector")
    print("="*60)

    if not ckpt_path.exists():
        print(f"Checkpoint not found: {ckpt_path}"); sys.exit(1)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(str(ckpt_path), map_location=device)
    print(f"\n  Checkpoint: epoch={ckpt['epoch']}  val_auc={ckpt['val_auc']:.4f}")

    from models.detector import build_detector
    model = build_detector(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    results = {}

    # ── 1. Standard evaluation on val and test ──────────────────────
    print("\n[1] Standard evaluation")

    print("  Running on val split...")
    val_probs, val_labels, val_clips, val_types = run_inference(
        model, device, cache_path, 'val', use_tta=False)
    val_std, val_per = compute_metrics(val_probs, val_labels, val_types)
    print(f"  Val  — AUC={val_std['auc']:.4f}  "
          f"F1={val_std['f1']:.4f}  Acc={val_std['acc']:.4f}")

    print("  Running on test split...")
    te_probs, te_labels, te_clips, te_types = run_inference(
        model, device, cache_path, 'test', use_tta=False)
    te_std, te_per = compute_metrics(te_probs, te_labels, te_types)
    print(f"  Test — AUC={te_std['auc']:.4f}  "
          f"F1={te_std['f1']:.4f}  Acc={te_std['acc']:.4f}")
    for mt, m in sorted(te_per.items()):
        print(f"    {mt:<22}: AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

    results['standard'] = {
        'val': {'overall': val_std, 'per_type': val_per},
        'test': {'overall': te_std, 'per_type': te_per},
    }

    # ── 2. Threshold tuning ─────────────────────────────────────────
    print("\n[2] Threshold tuning (from validation set)")
    best_thr, best_val_f1 = find_best_threshold(val_probs, val_labels)
    print(f"  Best threshold: {best_thr:.2f}  (val F1={best_val_f1:.4f} "
          f"vs default {val_std['f1']:.4f})")

    te_tuned, te_per_tuned = compute_metrics(
        te_probs, te_labels, te_types, threshold=best_thr)
    print(f"  Test (tuned thr={best_thr:.2f}) — "
          f"AUC={te_tuned['auc']:.4f}  "
          f"F1={te_tuned['f1']:.4f}  Acc={te_tuned['acc']:.4f}")
    for mt, m in sorted(te_per_tuned.items()):
        print(f"    {mt:<22}: AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

    results['threshold_tuned'] = {
        'threshold': best_thr,
        'test': {'overall': te_tuned, 'per_type': te_per_tuned},
    }

    # ── 3. TTA evaluation ───────────────────────────────────────────
    print("\n[3] Test-Time Augmentation (TTA)")
    print("  Running TTA on test split (takes 2x longer)...")
    te_tta_probs, te_tta_labels, te_tta_clips, te_tta_types = run_inference(
        model, device, cache_path, 'test', use_tta=True)
    te_tta, te_per_tta = compute_metrics(
        te_tta_probs, te_tta_labels, te_tta_types, threshold=best_thr)
    print(f"  Test (TTA + tuned thr) — "
          f"AUC={te_tta['auc']:.4f}  "
          f"F1={te_tta['f1']:.4f}  Acc={te_tta['acc']:.4f}")
    for mt, m in sorted(te_per_tta.items()):
        print(f"    {mt:<22}: AUC={m['auc']:.4f}  F1={m['f1']:.4f}")

    results['tta'] = {
        'threshold': best_thr,
        'test': {'overall': te_tta, 'per_type': te_per_tta},
    }

    # ── 4. Clip-level majority vote ─────────────────────────────────
    print("\n[4] Clip-level majority vote")
    clip_data = defaultdict(lambda: {'probs':[], 'label':None, 'type':None})
    for prob, label, clip, mt in zip(
            te_tta_probs, te_tta_labels, te_tta_clips, te_tta_types):
        clip_data[clip]['probs'].append(prob)
        clip_data[clip]['label'] = int(label)
        clip_data[clip]['type']  = mt

    clip_probs, clip_labels, clip_types = [], [], []
    for clip, d in clip_data.items():
        clip_probs.append(float(np.mean(d['probs'])))
        clip_labels.append(d['label'])
        clip_types.append(d['type'])

    clip_probs  = np.array(clip_probs)
    clip_labels = np.array(clip_labels)
    clip_mv, clip_per_mv = compute_metrics(
        clip_probs, clip_labels, clip_types, threshold=best_thr)
    print(f"  Clip-level (majority vote) — "
          f"AUC={clip_mv['auc']:.4f}  "
          f"F1={clip_mv['f1']:.4f}  Acc={clip_mv['acc']:.4f}")
    for mt, m in sorted(clip_per_mv.items()):
        print(f"    {mt:<22}: AUC={m['auc']:.4f}")

    results['clip_level'] = {
        'test': {'overall': clip_mv, 'per_type': clip_per_mv},
    }

    # ── 5. Save ROC curve data for plotting ─────────────────────────
    print("\n[5] Saving ROC curve data...")
    roc_data = {}
    fpr_all, tpr_all, _ = roc_curve(te_tta_labels, te_tta_probs)
    roc_data['overall'] = {
        'fpr': fpr_all.tolist(), 'tpr': tpr_all.tolist(),
        'auc': float(roc_auc_score(te_tta_labels, te_tta_probs))
    }
    for mt in sorted(set(te_tta_types)):
        if mt == 'real': continue
        idx = [i for i,t in enumerate(te_tta_types) if t in (mt,'real')]
        if len(idx) < 10: continue
        ml, mp = te_tta_labels[idx], te_tta_probs[idx]
        if len(np.unique(ml)) < 2: continue
        fpr, tpr, _ = roc_curve(ml, mp)
        roc_data[mt] = {
            'fpr': fpr.tolist(), 'tpr': tpr.tolist(),
            'auc': float(roc_auc_score(ml, mp))
        }
    results['roc_data'] = roc_data

    # ── 6. Confusion matrix ─────────────────────────────────────────
    te_preds = (te_tta_probs >= best_thr).astype(int)
    cm = confusion_matrix(te_tta_labels, te_preds)
    results['confusion_matrix'] = cm.tolist()
    tn, fp, fn, tp = cm.ravel()
    print(f"\n  Confusion matrix (test, TTA, thr={best_thr:.2f}):")
    print(f"    TN={tn}  FP={fp}  FN={fn}  TP={tp}")
    print(f"    Precision={tp/(tp+fp+1e-8):.4f}  "
          f"Recall={tp/(tp+fn+1e-8):.4f}")

    # ── 7. Save everything ──────────────────────────────────────────
    out_path = Path(model_dir) / "run_full_hybrid" / "final_results.json"
    with open(out_path, 'w') as fout:
        json.dump(results, fout, indent=2)
    print(f"\n  All results saved to: {out_path}")

    # ── 8. Print paper table ────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  PAPER RESULTS TABLE")
    print(f"{'='*60}")
    print(f"  Method              |  AUC   |  F1    |  Acc")
    print(f"  ------------------- | ------ | ------ | ------")
    print(f"  Standard (thr=0.5)  | "
          f"{te_std['auc']:.4f} | {te_std['f1']:.4f} | {te_std['acc']:.4f}")
    print(f"  + Thresh tuning     | "
          f"{te_tuned['auc']:.4f} | {te_tuned['f1']:.4f} | {te_tuned['acc']:.4f}")
    print(f"  + TTA               | "
          f"{te_tta['auc']:.4f} | {te_tta['f1']:.4f} | {te_tta['acc']:.4f}")
    print(f"  + Clip vote (final) | "
          f"{clip_mv['auc']:.4f} | {clip_mv['f1']:.4f} | {clip_mv['acc']:.4f}")
    print(f"\n  Per-manipulation (TTA + clip vote):")
    for mt, m in sorted(clip_per_mv.items()):
        print(f"    {mt:<22}: AUC={m['auc']:.4f}  "
              f"F1={m['f1']:.4f}  Acc={m['acc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()