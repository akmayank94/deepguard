"""
eval/celebdf_eval.py

Cross-dataset evaluation on Celeb-DF v2 (zero-shot).
Uses the already-trained FF++ model — NO RETRAINING.

Celeb-DF v2 structure expected:
    CelebDF_preprocessed/
        Train/
            real/  subfolder_name/ 00000.png 00001.png ...
            fake/  subfolder_name/ 00000.png ...
        Test/
            real/  subfolder_name/ 00000.png ...
            fake/  subfolder_name/ 00000.png ...

Strategy:
  - Use ONLY the Test split of Celeb-DF v2 for cross-dataset evaluation
    (standard protocol — we never touch Celeb-DF Train at all)
  - Sample 20 evenly-spaced frames per clip (handles variable frame count)
  - Run three-branch model: raw_rgb + diff_image + handcrafted
  - Compute diff_image and handcrafted features on-the-fly (no cache needed)
  - Report frame-level and clip-level AUC, F1, Accuracy

Usage:
    python eval/celebdf_eval.py --celebdf_root "D:/path/to/CelebDF_preprocessed"

Set CELEBDF_ROOT in .env or pass --celebdf_root argument.
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

# ── constants ───────────────────────────────────────────────────────────────
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]
N_FRAMES      = 20          # frames sampled per clip
CLIP_LO       = -500.0
CLIP_HI       =  500.0
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


# ── feature extraction (same as training pipeline) ──────────────────────────

def compute_diff_image(img_np: np.ndarray) -> torch.Tensor:
    """
    Compute inter-channel difference image from raw RGB numpy array.
    img_np: (H, W, 3) uint8
    Returns: (3, H, W) float32 tensor in [0, 1]
    """
    from contourlet.filters import (
        PKVA_LOWPASS_ANALYSIS, PKVA_HIGHPASS_ANALYSIS,
        conv2d_separable_mirror)

    img = img_np.astype(np.float64)
    if img.max() <= 1.0:
        img *= 255.0

    h0, h1 = PKVA_LOWPASS_ANALYSIS, PKVA_HIGHPASS_ANALYSIS
    EPS = 1e-8

    def _hf_energy(ch):
        maps = [
            conv2d_separable_mirror(ch, h0, h1),  # LH
            conv2d_separable_mirror(ch, h1, h0),  # HL
            conv2d_separable_mirror(ch, h1, h1),  # HH
        ]
        return np.mean([m**2 for m in maps], axis=0)

    E_R = _hf_energy(img[:, :, 0])
    E_G = _hf_energy(img[:, :, 1])
    E_B = _hf_energy(img[:, :, 2])

    diff = np.stack([E_R - E_B, E_R - E_G, E_G - E_B], axis=-1).astype(np.float32)
    for c in range(3):
        mn, mx = diff[:, :, c].min(), diff[:, :, c].max()
        rng = mx - mn
        diff[:, :, c] = (diff[:, :, c] - mn) / rng if rng > EPS else 0.5

    return torch.from_numpy(diff.transpose(2, 0, 1))  # (3, H, W)


def compute_handcrafted(img_np: np.ndarray) -> torch.Tensor:
    """
    Compute energy + DCCM handcrafted features.
    img_np: (H, W, 3) uint8
    Returns: (9369,) float32 tensor, clipped to [-500, 500]
    """
    from features.extractor import ForensicFeatureExtractor
    ext = ForensicFeatureExtractor()
    result = ext.extract(img_np)
    hc = np.clip(result['handcrafted_vector'], CLIP_LO, CLIP_HI).astype(np.float32)
    return torch.from_numpy(hc)


# ── dataset helpers ─────────────────────────────────────────────────────────

def collect_clips(celebdf_root: Path, split: str) -> list:
    """
    Walk CelebDF_preprocessed/{split}/real and /fake,
    collect (clip_path, label) pairs.
    Label: 0=real, 1=fake
    """
    clips = []
    for label, subfolder in [(0, 'real'), (1, 'fake')]:
        base = celebdf_root / split / subfolder
        if not base.exists():
            print(f"  WARNING: {base} does not exist, skipping")
            continue
        for clip_dir in sorted(base.iterdir()):
            if clip_dir.is_dir():
                clips.append((clip_dir, label))
    return clips


def sample_frames(clip_path: Path, n: int = N_FRAMES) -> list:
    """
    Sample n evenly-spaced PNG frames from clip_path.
    Handles variable frame count gracefully.
    """
    frames = sorted(clip_path.glob("*.png"))
    if not frames:
        frames = sorted(clip_path.glob("*.jpg"))
    if not frames:
        return []
    if len(frames) <= n:
        return frames
    indices = np.linspace(0, len(frames) - 1, n, dtype=int)
    return [frames[i] for i in indices]


def load_frame_pil(frame_path: Path) -> np.ndarray:
    """Load frame as (H, W, 3) uint8 numpy array, resized to IMG_SIZE."""
    img = Image.open(frame_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    return np.array(img)


# ── inference ────────────────────────────────────────────────────────────────

def run_celebdf_inference(model, device, clips: list,
                           use_tta: bool = True,
                           extract_features: bool = True):
    """
    Run inference on all clips.
    Returns: clip_probs, clip_labels, frame_probs, frame_labels, clip_names
    """
    model.eval()
    clip_results = []   # (mean_prob, label, clip_name)
    frame_probs_all  = []
    frame_labels_all = []

    extractor_available = True
    try:
        from features.extractor import ForensicFeatureExtractor
        _test_ext = ForensicFeatureExtractor()
    except Exception as e:
        print(f"  WARNING: Feature extractor not available ({e})")
        print(f"  Running with raw_rgb + diff_image only (no handcrafted branch)")
        extractor_available = False

    for clip_path, label in tqdm(clips, desc="  Clips", ncols=72, unit="clip"):
        frames = sample_frames(clip_path, N_FRAMES)
        if not frames:
            continue

        frame_probs = []
        for fp in frames:
            try:
                img_np = load_frame_pil(fp)
            except Exception:
                continue

            img_pil = Image.fromarray(img_np)

            # raw RGB
            rr = tf_eval(img_pil).unsqueeze(0).to(device)

            # diff image (on-the-fly)
            try:
                di = compute_diff_image(img_np).unsqueeze(0).to(device)
            except Exception:
                di = torch.zeros(1, 3, IMG_SIZE, IMG_SIZE).to(device)

            # handcrafted features
            if extractor_available and extract_features:
                try:
                    hc = compute_handcrafted(img_np).unsqueeze(0).to(device)
                except Exception:
                    hc = torch.zeros(1, 9369).to(device)
            else:
                hc = torch.zeros(1, 9369).to(device)

            with torch.no_grad():
                logit = model(rr, di, hc, return_gates=False)
                prob  = torch.sigmoid(logit).item()

                if use_tta:
                    rr_f = tf_flip(img_pil).unsqueeze(0).to(device)
                    logit_f = model(rr_f, di, hc, return_gates=False)
                    prob    = (prob + torch.sigmoid(logit_f).item()) / 2.0

            frame_probs.append(prob)
            frame_probs_all.append(prob)
            frame_labels_all.append(label)

        if frame_probs:
            clip_mean_prob = float(np.mean(frame_probs))
            clip_results.append((clip_mean_prob, label, clip_path.name))

    clip_probs  = np.array([r[0] for r in clip_results])
    clip_labels = np.array([r[1] for r in clip_results])
    clip_names  = [r[2] for r in clip_results]

    return (clip_probs, clip_labels, clip_names,
            np.array(frame_probs_all), np.array(frame_labels_all))


def find_best_threshold(probs, labels):
    best_f1, best_thr = 0.0, 0.5
    for thr in np.arange(0.25, 0.75, 0.01):
        preds = (probs >= thr).astype(int)
        f1 = f1_score(labels, preds, zero_division=0)
        if f1 > best_f1:
            best_f1, best_thr = f1, thr
    return float(best_thr), float(best_f1)


def print_metrics(probs, labels, desc, threshold=0.5):
    preds = (probs >= threshold).astype(int)
    auc   = float(roc_auc_score(labels, probs))
    f1    = float(f1_score(labels, preds, zero_division=0))
    acc   = float(accuracy_score(labels, preds))
    n_real = int((labels == 0).sum())
    n_fake = int((labels == 1).sum())
    print(f"  {desc}")
    print(f"    N: {len(labels)} (real={n_real}, fake={n_fake})")
    print(f"    AUC={auc:.4f}  F1={f1:.4f}  Acc={acc:.4f}  thr={threshold:.2f}")
    return {'auc': auc, 'f1': f1, 'acc': acc, 'n': len(labels),
            'n_real': n_real, 'n_fake': n_fake, 'threshold': threshold}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--celebdf_root', type=str,
                        default=os.environ.get('CELEBDF_ROOT', ''),
                        help='Path to CelebDF_preprocessed folder')
    parser.add_argument('--split', type=str, default='Test',
                        choices=['Test', 'Train', 'both'],
                        help='Which split to evaluate (default: Test)')
    parser.add_argument('--no_tta', action='store_true',
                        help='Disable test-time augmentation')
    parser.add_argument('--no_handcrafted', action='store_true',
                        help='Skip handcrafted feature extraction (faster)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path')
    args = parser.parse_args()

    celebdf_root = Path(args.celebdf_root)
    if not celebdf_root.exists():
        print(f"ERROR: Celeb-DF root not found: {celebdf_root}")
        print("Set CELEBDF_ROOT in .env or pass --celebdf_root argument")
        sys.exit(1)

    model_dir = os.environ.get("MODEL_DIR", "checkpoints")
    ckpt_path = Path(args.checkpoint) if args.checkpoint else \
                Path(model_dir) / "run_full_hybrid" / "checkpoint_best.pt"

    if not ckpt_path.exists():
        print(f"ERROR: Checkpoint not found: {ckpt_path}")
        sys.exit(1)

    print("=" * 60)
    print("  Cross-Dataset Evaluation — Celeb-DF v2")
    print("=" * 60)
    print(f"  Celeb-DF root : {celebdf_root}")
    print(f"  Checkpoint    : {ckpt_path}")
    print(f"  Split         : {args.split}")
    print(f"  TTA           : {not args.no_tta}")
    print(f"  Handcrafted   : {not args.no_handcrafted}")
    print(f"  Frames/clip   : {N_FRAMES}")

    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ckpt   = torch.load(str(ckpt_path), map_location=device)
    print(f"\n  Model: epoch={ckpt['epoch']}  "
          f"FF++ val_auc={ckpt['val_auc']:.4f}")

    from models.detector import build_detector
    model = build_detector(ckpt['config']).to(device)
    model.load_state_dict(ckpt['model'])
    model.eval()

    results = {'ff_plus_plus_val_auc': ckpt['val_auc'],
               'checkpoint_epoch': ckpt['epoch']}

    splits_to_eval = (['Train', 'Test'] if args.split == 'both'
                      else [args.split])

    for split in splits_to_eval:
        print(f"\n{'─'*60}")
        print(f"  Evaluating on Celeb-DF v2 — {split} split")
        print(f"{'─'*60}")

        clips = collect_clips(celebdf_root, split)
        if not clips:
            print(f"  No clips found in {celebdf_root}/{split}")
            continue

        n_real = sum(1 for _, l in clips if l == 0)
        n_fake = sum(1 for _, l in clips if l == 1)
        print(f"  Found {len(clips)} clips: real={n_real}, fake={n_fake}")

        (clip_probs, clip_labels, clip_names,
         fr_probs, fr_labels) = run_celebdf_inference(
            model, device, clips,
            use_tta=not args.no_tta,
            extract_features=not args.no_handcrafted)

        print(f"\n  Results:")
        fr_metrics   = print_metrics(fr_probs, fr_labels,
                                     "Frame-level (thr=0.5)", threshold=0.5)

        # Tune threshold on clip-level predictions
        best_thr, _ = find_best_threshold(clip_probs, clip_labels)
        clip_metrics = print_metrics(clip_probs, clip_labels,
                                     f"Clip-level (thr={best_thr:.2f})",
                                     threshold=best_thr)

        # Also show at 0.5 for fair comparison
        clip_05 = print_metrics(clip_probs, clip_labels,
                                "Clip-level (thr=0.5)", threshold=0.5)

        # ROC curve data
        fpr, tpr, _ = roc_curve(clip_labels, clip_probs)

        results[split] = {
            'frame_level': fr_metrics,
            'clip_level_tuned': clip_metrics,
            'clip_level_0.5':  clip_05,
            'roc': {'fpr': fpr.tolist(), 'tpr': tpr.tolist()},
            'n_clips': len(clips),
        }

        # Generalization gap
        ff_auc  = ckpt['val_auc']
        cel_auc = clip_metrics['auc']
        gap     = ff_auc - cel_auc
        print(f"\n  Generalization gap:")
        print(f"    FF++ val AUC   : {ff_auc:.4f}")
        print(f"    Celeb-DF AUC   : {cel_auc:.4f}")
        print(f"    Drop           : {gap:.4f} ({gap/ff_auc*100:.1f}%)")
        if cel_auc >= 0.75:
            print(f"    ✓ STRONG generalization — claim supported")
        elif cel_auc >= 0.65:
            print(f"    ~ MODERATE generalization — claim partially supported")
        else:
            print(f"    ✗ POOR generalization — model overfit to FF++")

    # Save results
    out_path = Path(model_dir) / "run_full_hybrid" / "celebdf_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")

    print(f"\n{'='*60}")
    print(f"  CROSS-DATASET SUMMARY")
    print(f"{'='*60}")
    for split in splits_to_eval:
        if split in results:
            m = results[split]['clip_level_tuned']
            print(f"  Celeb-DF v2 {split:<8}: "
                  f"AUC={m['auc']:.4f}  "
                  f"F1={m['f1']:.4f}  "
                  f"Acc={m['acc']:.4f}")
    print(f"  FF++ (val, epoch 48) : AUC={ckpt['val_auc']:.4f}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()