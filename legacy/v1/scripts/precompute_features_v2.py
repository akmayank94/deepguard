"""
scripts/precompute_features_v2.py

Precomputes ALL features for the new FF++C23_preprocessed dataset:
  1. handcrafted  — CT energy + DCCM (9369-dim) from frames_aligned/
  2. diff_image   — inter-channel energy diff map (224x224x3) from frames_aligned/
  3. prnu         — noise residual features (30-dim) from frames_raw/
  4. rppg         — heartbeat frequency features (32-dim) from rppg.npy
  5. frame_paths  — stored so dataset can load aligned frames for raw RGB branch

HDF5 structure:
  cache/features_v2_{mode}_seed42.h5
    train/
      handcrafted   (N, 9369)  float32
      diff_images   (N, 224, 224, 3)  float32
      prnu          (N, 30)    float32
      rppg          (N, 32)    float32
      labels        (N,)       int8
      clip_names    (N,)       bytes
      manip_types   (N,)       bytes
      frame_indices (N,)       int32  — which of the 20 aligned frames this is
    val/  ...
    test/ ...

Usage:
    python scripts/precompute_features_v2.py
    python scripts/precompute_features_v2.py --pilot
"""

import sys, os, argparse, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import numpy as np
import h5py
import cv2
from PIL import Image
from tqdm import tqdm

from data.dataset_ff_v2 import build_ff_v2_splits
from features.extractor import ForensicFeatureExtractor
from features.prnu      import extract_prnu_features
from features.rppg      import extract_rppg_features

FRAMES_PER_CLIP = 20   # aligned frames per clip
NEW_DATA_ROOT   = os.environ.get(
    "DATA_ROOT_V2",
    "D:/DO NOT TOUCH/deepfake_microexpression/data/FF++C23_preprocessed")
CACHE_DIR       = os.environ.get("CACHE_DIR", "cache")


def _load_aligned_frame(clip_path: Path, frame_idx: int) -> np.ndarray:
    """Load one aligned 224x224 JPG → numpy RGB uint8."""
    jpg = clip_path / "frames_aligned" / f"{frame_idx:03d}.jpg"
    img = Image.open(str(jpg)).convert("RGB")
    return np.array(img)


def _compute_handcrafted_and_diff(img_rgb: np.ndarray,
                                   extractor: ForensicFeatureExtractor):
    """Returns (handcrafted_9369, diff_image_224x224x3)."""
    result = extractor.extract(img_rgb)
    hc     = np.clip(result['handcrafted_vector'], -500, 500).astype(np.float32)
    di     = result['diff_image'].astype(np.float32)
    return hc, di


def process_split(entries, split_name, hf_group, extractor, pilot):
    """Process one split and write to HDF5 group."""
    # Each entry becomes FRAMES_PER_CLIP rows in the HDF5
    n_clips  = len(entries)
    n_frames = n_clips * FRAMES_PER_CLIP

    print(f"\n  Processing {split_name} split ({n_clips} clips → {n_frames} frames)...")

    # Pre-allocate HDF5 datasets
    hf_group.create_dataset('handcrafted',   (n_frames, 9369),        dtype='float32',
                            chunks=(32, 9369),        compression='gzip', compression_opts=4)
    hf_group.create_dataset('diff_images',   (n_frames, 224, 224, 3), dtype='float32',
                            chunks=(4, 224, 224, 3),  compression='gzip', compression_opts=4)
    hf_group.create_dataset('prnu',          (n_frames, 30),          dtype='float32',
                            chunks=(32, 30),           compression='gzip', compression_opts=4)
    hf_group.create_dataset('rppg',          (n_frames, 32),          dtype='float32',
                            chunks=(32, 32),           compression='gzip', compression_opts=4)
    hf_group.create_dataset('labels',        (n_frames,),             dtype='int8')
    hf_group.create_dataset('frame_indices', (n_frames,),             dtype='int32')
    dt = h5py.special_dtype(vlen=bytes)
    hf_group.create_dataset('clip_names',    (n_frames,),             dtype=dt)
    hf_group.create_dataset('manip_types',   (n_frames,),             dtype=dt)

    row    = 0
    errors = 0

    for entry in tqdm(entries, desc=f"  {split_name}", ncols=72, unit="clip"):
        clip_path  = entry['path']
        label      = entry['label']
        manip_type = entry['manip_type']
        clip_name  = entry['clip_name']

        # PRNU: from frames_raw/ (computed once per clip, shared across all frames)
        raw_pngs = sorted((clip_path / "frames_raw").glob("*.png"))
        prnu_feat = extract_prnu_features(raw_pngs, n_frames=20)

        # rPPG: from rppg.npy (computed once per clip)
        rppg_path = clip_path / "rppg.npy"
        rppg_feat = extract_rppg_features(rppg_path)

        # Per-frame features (handcrafted + diff_image) from aligned frames
        for fi in range(FRAMES_PER_CLIP):
            try:
                img_rgb = _load_aligned_frame(clip_path, fi)
                hc, di  = _compute_handcrafted_and_diff(img_rgb, extractor)
            except Exception as e:
                errors += 1
                # Use zeros for failed frames
                hc = np.zeros(9369, dtype=np.float32)
                di = np.zeros((224, 224, 3), dtype=np.float32)

            hf_group['handcrafted'][row]   = hc
            hf_group['diff_images'][row]   = di
            hf_group['prnu'][row]          = prnu_feat    # same for all frames of clip
            hf_group['rppg'][row]          = rppg_feat    # same for all frames of clip
            hf_group['labels'][row]        = label
            hf_group['frame_indices'][row] = fi
            hf_group['clip_names'][row]    = clip_name.encode()
            hf_group['manip_types'][row]   = manip_type.encode()
            row += 1

    print(f"  {split_name}: {row} frames written, {errors} errors")
    return errors


def verify_split(hf_group, split_name):
    """Quick sanity check on written data."""
    n      = hf_group['labels'].shape[0]
    labels = hf_group['labels'][:]
    real_n = int((labels == 0).sum())
    fake_n = int((labels == 1).sum())
    ratio  = fake_n / max(real_n, 1)

    hc_sample = hf_group['handcrafted'][:10]
    di_sample = hf_group['diff_images'][:4]

    has_nan_hc = bool(np.isnan(hc_sample).any() or np.isinf(hc_sample).any())
    has_nan_di = bool(np.isnan(di_sample).any() or np.isinf(di_sample).any())
    di_in_01   = bool(di_sample.min() >= 0.0 and di_sample.max() <= 1.0)

    prnu_sample = hf_group['prnu'][:10]
    rppg_sample = hf_group['rppg'][:10]
    has_nan_pr  = bool(np.isnan(prnu_sample).any())
    has_nan_rp  = bool(np.isnan(rppg_sample).any())

    print(f"  {split_name}: frames={n}  real={real_n}  fake={fake_n}  ratio={ratio:.2f}")
    print(f"    handcrafted NaN: {has_nan_hc}  diff_image [0,1]: {di_in_01}")
    print(f"    prnu NaN: {has_nan_pr}  rppg NaN: {has_nan_rp}")
    print(f"    prnu sample mean={prnu_sample.mean():.4f}  "
          f"rppg sample mean={rppg_sample.mean():.4f}")

    if ratio > 1.5 or ratio < 0.5:
        print(f"    WARNING: imbalanced split (ratio={ratio:.2f})")
    if has_nan_hc or has_nan_di or has_nan_pr or has_nan_rp:
        print(f"    WARNING: NaN values detected")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true', help='Use 100 clips per type')
    args = p.parse_args()

    mode       = "pilot" if args.pilot else "full"
    cache_path = Path(CACHE_DIR) / f"features_v2_{mode}_seed42.h5"
    Path(CACHE_DIR).mkdir(exist_ok=True)

    print("="*60)
    print(f"  Feature Precomputation v2 — {mode.upper()} mode")
    print("="*60)
    print(f"  Data root  : {NEW_DATA_ROOT}")
    print(f"  Cache path : {cache_path}")

    if cache_path.exists():
        print(f"  WARNING: Cache already exists — will overwrite")
        cache_path.unlink()

    print(f"\n  Building dataset splits...")
    train_entries, val_entries, test_entries = build_ff_v2_splits(
        NEW_DATA_ROOT, pilot=args.pilot)

    print(f"\n  Initializing feature extractor...")
    extractor = ForensicFeatureExtractor()

    t0     = time.time()
    errors = 0

    with h5py.File(str(cache_path), 'w') as hf:
        for split_name, entries in [
            ('train', train_entries),
            ('val',   val_entries),
            ('test',  test_entries),
        ]:
            grp    = hf.create_group(split_name)
            errs   = process_split(entries, split_name, grp, extractor,
                                   args.pilot)
            errors += errs

    elapsed = time.time() - t0
    size_gb = cache_path.stat().st_size / 1e9

    print(f"\n{'='*60}")
    print(f"  Precomputation Complete")
    print(f"{'='*60}")
    print(f"  PASS  Cache saved : {cache_path}")
    print(f"  INFO  Total time  : {elapsed/60:.1f} min")
    print(f"  INFO  Cache size  : {size_gb:.1f} GB")
    print(f"  INFO  Total errors: {errors}")

    print(f"\nCache Verification")
    with h5py.File(str(cache_path), 'r') as hf:
        for split in ['train', 'val', 'test']:
            verify_split(hf[split], split)

    print(f"\n  Cache ready. Next step:")
    print(f"  python main_v2.py {'--pilot' if args.pilot else ''}")


if __name__ == "__main__":
    main()