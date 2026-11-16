"""
eval/evaluator.py

Post-training evaluation — loads best checkpoint and computes full
results table matching the paper's reporting format:

    Dataset | Method | Acc | F1 | AUC
    per manipulation type + overall

Also computes clip-level majority vote results (more robust than
frame-level averaging).
"""

import sys
import torch
import numpy as np
from pathlib import Path
from collections import defaultdict
from sklearn.metrics import (
    roc_auc_score, f1_score, accuracy_score,
    roc_curve, confusion_matrix,
)

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.detector import build_detector
from data.dataset_cached import CachedFeatureDataset, build_cached_dataloaders


def evaluate_full(
    checkpoint_path: str,
    cache_path: str,
    device: torch.device = None,
    split: str = 'test',
) -> dict:
    """
    Load checkpoint and run full evaluation on a cache split.

    Args:
        checkpoint_path: path to .pt checkpoint file
        cache_path:      path to HDF5 cache file
        device:          torch device (auto-detected if None)
        split:           which cache split to evaluate ('test', 'val')

    Returns:
        results dict with frame-level and clip-level metrics.
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    config = ckpt.get('config', {})

    model = build_detector(config).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    print(f"  Loaded checkpoint (epoch={ckpt.get('epoch','?')}, "
          f"val_auc={ckpt.get('val_auc', 0):.4f})")

    # Dataset
    dataset = CachedFeatureDataset(cache_path, split, augment_diff_image=False)
    loader  = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=0)

    all_logits     = []
    all_labels     = []
    all_manip_types= []
    all_clip_names = []
    all_frame_idxs = []

    with torch.no_grad():
        for batch in loader:
            diff_img    = batch['diff_image'].to(device)
            handcrafted = batch['handcrafted'].to(device)
            logits = model(diff_img, handcrafted, return_gates=False)
            all_logits.extend(logits.cpu().numpy())
            all_labels.extend(batch['label'].numpy())
            all_manip_types.extend(batch['manip_type'])
            all_clip_names.extend(batch['clip_name'])
            all_frame_idxs.extend(batch['frame_idx'] if isinstance(
                batch['frame_idx'], list) else batch['frame_idx'].numpy())

    all_logits = np.array(all_logits)
    all_labels = np.array(all_labels)
    all_probs  = 1 / (1 + np.exp(-all_logits))   # sigmoid
    all_preds  = (all_probs >= 0.5).astype(int)

    # --- Frame-level metrics ---
    frame_metrics = _compute_metrics(all_labels, all_probs, all_preds)

    # --- Per-manipulation-type frame metrics ---
    type_metrics = {}
    for mt in set(all_manip_types):
        if mt == 'real':
            continue
        idx = [i for i, t in enumerate(all_manip_types) if t in (mt, 'real')]
        if len(idx) < 10:
            continue
        mt_labels = all_labels[idx]
        mt_probs  = all_probs[idx]
        mt_preds  = all_preds[idx]
        if len(np.unique(mt_labels)) < 2:
            continue
        type_metrics[mt] = _compute_metrics(mt_labels, mt_probs, mt_preds)

    # --- Clip-level majority vote ---
    clip_results = defaultdict(lambda: {'probs': [], 'label': None, 'manip': None})
    for i, clip in enumerate(all_clip_names):
        if isinstance(clip, bytes):
            clip = clip.decode()
        clip_results[clip]['probs'].append(all_probs[i])
        clip_results[clip]['label']  = int(all_labels[i])
        mt = all_manip_types[i]
        if isinstance(mt, bytes):
            mt = mt.decode()
        clip_results[clip]['manip'] = mt

    clip_labels = []
    clip_probs  = []
    for clip, data in clip_results.items():
        clip_labels.append(data['label'])
        clip_probs.append(np.mean(data['probs']))

    clip_labels = np.array(clip_labels)
    clip_probs  = np.array(clip_probs)
    clip_preds  = (clip_probs >= 0.5).astype(int)
    clip_metrics = _compute_metrics(clip_labels, clip_probs, clip_preds)

    # --- Print results table ---
    _print_results(frame_metrics, clip_metrics, type_metrics)

    return {
        'frame':    frame_metrics,
        'clip':     clip_metrics,
        'by_type':  type_metrics,
        'arrays': {
            'labels': all_labels,
            'probs':  all_probs,
            'preds':  all_preds,
        }
    }


def _compute_metrics(labels, probs, preds) -> dict:
    return {
        'acc': float(accuracy_score(labels, preds)),
        'f1':  float(f1_score(labels, preds, zero_division=0)),
        'auc': float(roc_auc_score(labels, probs)),
        'n':   int(len(labels)),
    }


def _print_results(frame, clip, by_type):
    print(f"\n{'='*60}")
    print(f"  Evaluation Results")
    print(f"{'='*60}")
    print(f"  {'Level':<12}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}  {'N':>6}")
    print(f"  {'-'*50}")
    print(f"  {'Frame':<12}  "
          f"{frame['acc']*100:>6.2f}%  "
          f"{frame['f1']*100:>6.2f}%  "
          f"{frame['auc']*100:>6.2f}%  "
          f"{frame['n']:>6}")
    print(f"  {'Clip (vote)':<12}  "
          f"{clip['acc']*100:>6.2f}%  "
          f"{clip['f1']*100:>6.2f}%  "
          f"{clip['auc']*100:>6.2f}%  "
          f"{clip['n']:>6}")
    if by_type:
        print(f"\n  Per-manipulation-type (frame level):")
        print(f"  {'Type':<22}  {'Acc':>7}  {'F1':>7}  {'AUC':>7}")
        print(f"  {'-'*50}")
        for mt, m in sorted(by_type.items()):
            print(f"  {mt:<22}  "
                  f"{m['acc']*100:>6.2f}%  "
                  f"{m['f1']*100:>6.2f}%  "
                  f"{m['auc']*100:>6.2f}%")
    print(f"{'='*60}")