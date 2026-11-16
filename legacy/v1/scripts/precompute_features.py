"""
scripts/precompute_features.py

Offline feature precomputation — runs CT decomposition + energy + DCCM
on every frame in the dataset and saves results to an HDF5 cache file.

WHY THIS EXISTS:
    CT decomposition takes ~60-170ms per frame (CPU). During training, if we
    ran CT live on every batch it would completely bottleneck the GPU — the
    RTX 4050 would sit idle waiting for CPU feature computation. By precomputing
    once and caching to HDF5, training becomes a fast GPU-only operation.

    Pilot (200 clips × 20 frames = 4000 frames):
        ~4000 × 170ms ≈ 11 minutes one-time cost → then instant every epoch.
    Full dataset (5943 clips × 20 frames = 118,860 frames):
        ~118,860 × 170ms ≈ 5-6 hours one-time cost → then instant every epoch.

WHAT GETS SAVED PER FRAME:
    handcrafted_vector : float32 (3240,)  — MLP branch input
    diff_image         : float32 (224, 224, 3) — EfficientNet branch input
    energy_vector      : float32 (3048,)  — energy component (for ablation)
    dccm_vector        : float32 (192,)   — DCCM component (for ablation)
    label              : int8             — 0=real, 1=fake
    manip_type         : str              — 'real','Deepfakes',etc.
    clip_name          : str              — e.g. 'real_000', 'Deepfakes_274_412'
    frame_idx          : int8             — 0-19

HDF5 STRUCTURE:
    cache.h5
    ├── train/
    │   ├── handcrafted   (N_train, 3240) float32
    │   ├── diff_images   (N_train, 224, 224, 3) float32
    │   ├── energy        (N_train, 3048) float32
    │   ├── dccm          (N_train, 192)  float32
    │   ├── labels        (N_train,)      int8
    │   ├── manip_types   (N_train,)      h5py string
    │   ├── clip_names    (N_train,)      h5py string
    │   └── frame_indices (N_train,)      int8
    ├── val/   (same structure)
    └── test/  (same structure)

Usage:
    # Pilot run (200 sequences — do this first)
    python scripts/precompute_features.py --pilot

    # Full dataset
    python scripts/precompute_features.py

    # Force recompute even if cache exists
    python scripts/precompute_features.py --pilot --force

    # Custom paths
    python scripts/precompute_features.py --pilot \
        --data_root "D:/data/deepfakes_preprocessed" \
        --cache_dir "D:/cache/spectral"
"""

import os
import sys
import time
import argparse
import numpy as np
from pathlib import Path
from typing import Optional

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv(override=True)

import h5py
from PIL import Image as PILImage
from tqdm import tqdm

from contourlet import ContourletTransform
from features.extractor import ForensicFeatureExtractor
from data.dataset_ff import (
    build_ff_datasets,
    MANIP_TYPES,
    LABEL_REAL,
    LABEL_FAKE,
)


# ---------------------------------------------------------------------------
# ANSI colours
# ---------------------------------------------------------------------------
GREEN  = "\033[92m"
RED    = "\033[91m"
YELLOW = "\033[93m"
CYAN   = "\033[96m"
RESET  = "\033[0m"
BOLD   = "\033[1m"

def _info(msg):  print(f"  {CYAN}INFO{RESET}  {msg}")
def _pass(msg):  print(f"  {GREEN}PASS{RESET}  {msg}")
def _warn(msg):  print(f"  {YELLOW}WARN{RESET}  {msg}")
def _fail(msg):  print(f"  {RED}FAIL{RESET}  {msg}")


# ---------------------------------------------------------------------------
# Feature dimensions — determined at runtime from one test extraction
# ---------------------------------------------------------------------------

def _probe_feature_dims(extractor: ForensicFeatureExtractor) -> dict:
    """
    Run one extraction on a dummy image to get actual feature dimensions.
    This is more reliable than hardcoding since dims depend on CT subband sizes.
    """
    dummy = np.random.randint(0, 256, (224, 224, 3), dtype=np.uint8)
    result = extractor.extract(dummy)
    return {
        'handcrafted': len(result['handcrafted_vector']),
        'energy':      len(result['energy_vector']),
        'dccm':        len(result['dccm_vector']),
        'diff_image':  result['diff_image'].shape,   # (224, 224, 3)
    }


# ---------------------------------------------------------------------------
# HDF5 cache writer
# ---------------------------------------------------------------------------

class CacheWriter:
    """
    Manages creation and writing of the HDF5 feature cache.

    Uses chunked storage with gzip compression for diff_images (large)
    and uncompressed storage for feature vectors (fast random access
    during training DataLoader).
    """

    def __init__(self, cache_path: Path, dims: dict, split_sizes: dict):
        """
        Args:
            cache_path:   path to .h5 file to create
            dims:         dict from _probe_feature_dims()
            split_sizes:  dict {'train': N, 'val': N, 'test': N} — frame counts
        """
        self.cache_path  = cache_path
        self.dims        = dims
        self.split_sizes = split_sizes
        self._file       = None
        self._cursors    = {'train': 0, 'val': 0, 'test': 0}

    def __enter__(self):
        self._file = h5py.File(self.cache_path, 'w')
        self._create_datasets()
        return self

    def __exit__(self, *args):
        if self._file:
            self._file.close()

    def _create_datasets(self):
        """Pre-allocate all HDF5 datasets with known sizes."""
        H, W, C = self.dims['diff_image']

        for split, N in self.split_sizes.items():
            grp = self._file.create_group(split)

            # Feature vectors — uncompressed for fast random access
            grp.create_dataset('handcrafted',
                shape=(N, self.dims['handcrafted']),
                dtype='float32', chunks=(min(64, N), self.dims['handcrafted']))

            grp.create_dataset('energy',
                shape=(N, self.dims['energy']),
                dtype='float32', chunks=(min(64, N), self.dims['energy']))

            grp.create_dataset('dccm',
                shape=(N, self.dims['dccm']),
                dtype='float32', chunks=(min(64, N), self.dims['dccm']))

            # Diff images — gzip compressed (large, written once, read sequentially)
            grp.create_dataset('diff_images',
                shape=(N, H, W, C),
                dtype='float32',
                chunks=(min(16, N), H, W, C),
                compression='gzip', compression_opts=4)

            # Labels and metadata
            grp.create_dataset('labels',
                shape=(N,), dtype='int8')

            grp.create_dataset('frame_indices',
                shape=(N,), dtype='int8')

            # Variable-length strings for clip names and manip types
            str_dtype = h5py.special_dtype(vlen=str)
            grp.create_dataset('clip_names',
                shape=(N,), dtype=str_dtype)
            grp.create_dataset('manip_types',
                shape=(N,), dtype=str_dtype)

    def write_frame(self, split: str, result: dict, label: int,
                    clip_name: str, manip_type: str, frame_idx: int):
        """
        Write one frame's features to the cache.

        Args:
            split:      'train', 'val', or 'test'
            result:     dict from ForensicFeatureExtractor.extract()
            label:      0 or 1
            clip_name:  clip folder name
            manip_type: manipulation type string
            frame_idx:  0-19
        """
        i = self._cursors[split]
        grp = self._file[split]

        grp['handcrafted'][i]    = result['handcrafted_vector']
        grp['energy'][i]         = result['energy_vector']
        grp['dccm'][i]           = result['dccm_vector']
        grp['diff_images'][i]    = result['diff_image']
        grp['labels'][i]         = label
        grp['frame_indices'][i]  = frame_idx
        grp['clip_names'][i]     = clip_name
        grp['manip_types'][i]    = manip_type

        self._cursors[split] += 1


# ---------------------------------------------------------------------------
# Core precomputation function
# ---------------------------------------------------------------------------

def precompute(
    data_root: str,
    cache_dir: str,
    pilot: bool = True,
    force: bool = False,
    seed: int = 42,
) -> Path:
    """
    Run full precomputation pipeline and save HDF5 cache.

    Args:
        data_root: path to deepfakes_preprocessed/
        cache_dir: directory to save cache files
        pilot:     if True, use 200-sequence pilot subset
        force:     if True, recompute even if cache exists
        seed:      random seed (must match dataset split seed)

    Returns:
        Path to the created .h5 cache file.
    """
    cache_dir  = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    mode_tag   = "pilot" if pilot else "full"
    cache_path = cache_dir / f"features_{mode_tag}_seed{seed}.h5"

    # --- Check if cache already exists ---
    if cache_path.exists() and not force:
        _info(f"Cache already exists: {cache_path}")
        _info(f"Use --force to recompute. Skipping.")
        return cache_path

    print(f"\n{'='*60}")
    print(f"{BOLD}  Feature Precomputation — {mode_tag.upper()} mode{RESET}")
    print(f"{'='*60}")
    _info(f"Data root  : {data_root}")
    _info(f"Cache path : {cache_path}")
    _info(f"Pilot mode : {pilot}")

    # --- Build datasets (clip-level splits, no transforms needed here) ---
    _info("Building dataset splits...")
    train_ds, val_ds, test_ds = build_ff_datasets(
        data_root=data_root,
        pilot=pilot,
        seed=seed,
        return_path=True,   # need image paths for direct loading
    )
    _info(f"Split sizes — train:{len(train_ds)}  val:{len(val_ds)}  test:{len(test_ds)}")

    split_sizes = {
        'train': len(train_ds),
        'val':   len(val_ds),
        'test':  len(test_ds),
    }

    # --- Build shared extractor (single CT instance reused for all frames) ---
    _info("Initializing feature extractor (shared CT instance)...")
    ct        = ContourletTransform(num_levels=2, directions_per_level=[4, 8])
    extractor = ForensicFeatureExtractor(ct=ct)

    # Probe actual feature dimensions from one dummy extraction
    dims = _probe_feature_dims(extractor)
    _info(f"Feature dims — handcrafted:{dims['handcrafted']}  "
          f"energy:{dims['energy']}  dccm:{dims['dccm']}  "
          f"diff_image:{dims['diff_image']}")

    # --- Precompute and write ---
    total_frames = sum(split_sizes.values())
    _info(f"Processing {total_frames} frames total...")

    t_start = time.time()
    errors  = []

    with CacheWriter(cache_path, dims, split_sizes) as writer:
        for split_name, dataset in [
            ('train', train_ds),
            ('val',   val_ds),
            ('test',  test_ds),
        ]:
            print(f"\n  Processing {split_name} split "
                  f"({len(dataset)} frames)...")

            for item in tqdm(dataset, desc=f"  {split_name}", ncols=70,
                             unit="frame"):
                img_path   = item['image_path']
                label      = int(item['label'])
                clip_name  = item['clip_name']
                manip_type = item['manip_type']
                frame_idx  = int(item['frame_idx'])

                try:
                    img    = np.array(PILImage.open(img_path).convert("RGB"))
                    result = extractor.extract(img)
                    writer.write_frame(
                        split=split_name,
                        result=result,
                        label=label,
                        clip_name=clip_name,
                        manip_type=manip_type,
                        frame_idx=frame_idx,
                    )
                except Exception as e:
                    errors.append((str(img_path), str(e)))
                    _warn(f"Failed: {img_path}: {e}")
                    # Write zeros so HDF5 indices stay aligned
                    writer.write_frame(
                        split=split_name,
                        result={
                            'handcrafted_vector': np.zeros(dims['handcrafted'], np.float32),
                            'energy_vector':      np.zeros(dims['energy'],      np.float32),
                            'dccm_vector':        np.zeros(dims['dccm'],        np.float32),
                            'diff_image':         np.zeros(dims['diff_image'],  np.float32),
                        },
                        label=label,
                        clip_name=clip_name,
                        manip_type=manip_type,
                        frame_idx=frame_idx,
                    )

    elapsed = time.time() - t_start

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"{BOLD}  Precomputation Complete{RESET}")
    print(f"{'='*60}")
    _pass(f"Cache saved : {cache_path}")
    _info(f"Total time  : {elapsed/60:.1f} min  "
          f"({elapsed/total_frames*1000:.0f} ms/frame avg)")
    _info(f"Cache size  : {cache_path.stat().st_size / 1e6:.1f} MB")

    if errors:
        _warn(f"{len(errors)} frames failed (written as zeros):")
        for path, err in errors[:5]:
            print(f"    {Path(path).name}: {err}")
    else:
        _pass(f"Zero errors — all {total_frames} frames processed successfully")

    return cache_path


# ---------------------------------------------------------------------------
# Cache verification
# ---------------------------------------------------------------------------

def verify_cache(cache_path: Path) -> bool:
    """
    Verify the HDF5 cache is well-formed and spot-check a few entries.

    Checks:
        - All expected datasets present in all splits
        - Shapes match expected dimensions
        - No NaN or Inf values in a random sample
        - Label distribution is approximately balanced
        - Diff images are in [0, 1] range

    Returns:
        True if all checks pass.
    """
    print(f"\n{BOLD}Cache Verification{RESET}")
    ok = True

    with h5py.File(cache_path, 'r') as f:
        required_keys = [
            'handcrafted', 'energy', 'dccm', 'diff_images',
            'labels', 'frame_indices', 'clip_names', 'manip_types'
        ]

        for split in ('train', 'val', 'test'):
            if split not in f:
                _fail(f"Split '{split}' missing from cache")
                ok = False
                continue

            grp = f[split]
            N   = grp['labels'].shape[0]
            _info(f"Split '{split}': {N} frames")

            # Check all keys present
            for key in required_keys:
                if key not in grp:
                    _fail(f"  Key '{key}' missing from {split}")
                    ok = False

            # Shape checks
            hc_shape = grp['handcrafted'].shape
            di_shape = grp['diff_images'].shape
            _info(f"  handcrafted shape : {hc_shape}")
            _info(f"  diff_images shape : {di_shape}")

            # Label distribution
            labels = grp['labels'][:]
            n_real = int((labels == 0).sum())
            n_fake = int((labels == 1).sum())
            _info(f"  Labels — real:{n_real}  fake:{n_fake}  "
                  f"ratio:{n_real/(n_fake+1e-6):.2f}")
            if abs(n_real - n_fake) / (n_real + n_fake + 1e-6) < 0.15:
                _pass(f"  Label balance is good (within 15% of 50/50)")
            else:
                _warn(f"  Label imbalance detected")

            # NaN/Inf check on random sample of 20 frames
            np.random.seed(0)
            sample_idx = np.random.choice(N, min(20, N), replace=False)
            hc_sample  = grp['handcrafted'][sorted(sample_idx)]
            if not np.any(np.isnan(hc_sample)) and not np.any(np.isinf(hc_sample)):
                _pass(f"  No NaN/Inf in handcrafted sample")
            else:
                _fail(f"  NaN or Inf found in handcrafted features!")
                ok = False

            # Diff image range check
            di_sample = grp['diff_images'][sorted(sample_idx[:4])]
            di_min, di_max = di_sample.min(), di_sample.max()
            if -0.01 <= di_min and di_max <= 1.01:
                _pass(f"  Diff images in [0,1]: min={di_min:.3f} max={di_max:.3f}")
            else:
                _warn(f"  Diff images out of range: [{di_min:.3f}, {di_max:.3f}]")

            # Spot-check one entry
            idx = 0
            cn  = grp['clip_names'][idx]
            mt  = grp['manip_types'][idx]
            lb  = grp['labels'][idx]
            fi  = grp['frame_indices'][idx]
            _info(f"  Entry[0]: clip={cn}  type={mt}  label={lb}  frame={fi}")

    return ok


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Precompute CT+energy+DCCM features and save to HDF5 cache")
    parser.add_argument(
        '--pilot', action='store_true', default=False,
        help='Use 200-sequence pilot subset (default: full dataset)')
    parser.add_argument(
        '--force', action='store_true', default=False,
        help='Recompute even if cache already exists')
    parser.add_argument(
        '--data_root', type=str, default=None,
        help='Path to deepfakes_preprocessed/. Overrides .env DATA_ROOT.')
    parser.add_argument(
        '--cache_dir', type=str, default=None,
        help='Directory to save cache. Overrides .env CACHE_DIR.')
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed (must match training seed). Default: 42.')
    parser.add_argument(
        '--verify_only', action='store_true', default=False,
        help='Skip computation, only verify an existing cache.')
    args = parser.parse_args()

    # Resolve paths
    data_root = args.data_root or os.environ.get("DATA_ROOT", "")
    cache_dir = args.cache_dir or os.environ.get("CACHE_DIR", "")

    if not data_root:
        print(f"{RED}ERROR:{RESET} DATA_ROOT not set. "
              f"Add to .env or pass --data_root")
        sys.exit(1)
    if not cache_dir:
        # Default: cache/ folder inside project root
        cache_dir = str(Path(__file__).parent.parent / "cache")
        _warn(f"CACHE_DIR not set — using default: {cache_dir}")

    mode_tag   = "pilot" if args.pilot else "full"
    cache_path = Path(cache_dir) / f"features_{mode_tag}_seed{args.seed}.h5"

    if args.verify_only:
        if not cache_path.exists():
            _fail(f"Cache not found: {cache_path}")
            sys.exit(1)
        ok = verify_cache(cache_path)
        sys.exit(0 if ok else 1)

    # Run precomputation
    cache_path = precompute(
        data_root=data_root,
        cache_dir=cache_dir,
        pilot=args.pilot,
        force=args.force,
        seed=args.seed,
    )

    # Always verify after writing
    ok = verify_cache(cache_path)
    if ok:
        print(f"\n  {GREEN}{BOLD}Cache is ready for training.{RESET}")
        print(f"  Next step: python scripts/precompute_features.py --pilot (if not done)")
        print(f"  Then     : python main.py --pilot\n")
    else:
        print(f"\n  {RED}Cache verification failed. Check errors above.{RESET}\n")
        sys.exit(1)


if __name__ == "__main__":
    main()