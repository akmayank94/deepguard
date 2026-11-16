"""
scripts/precompute_features_v3.py

Builds features_v3 cache by:
  1. Copying existing v2 features (handcrafted, diff_images, prnu, rppg, labels etc.)
  2. Adding NEW features: temporal_std (224,224,3), fdi (42,), dct (48,)

New Branch C input = prnu (30) + dct (48) = 78-dim combined forensic MLP.
(Concatenation happens in the dataset loader, not here.)

Usage:
    python scripts/precompute_features_v3.py --pilot
    python scripts/precompute_features_v3.py
"""

import sys, os, argparse, time
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import numpy as np
import h5py
from collections import defaultdict
from tqdm import tqdm

from features.temporal_consistency import extract_temporal_consistency
from features.facial_dynamics       import extract_fdi_features
from features.dct_stats             import extract_dct_features

DATA_ROOT_V2 = os.environ.get(
    "DATA_ROOT_V2",
    "D:/DO NOT TOUCH/deepfake_microexpression/data/FF++C23_preprocessed")
CACHE_DIR = os.environ.get("CACHE_DIR", "cache")


def process_split(src_grp, dst_grp, data_root, split_name):
    n = src_grp['labels'].shape[0]
    print(f"\n  {split_name} ({n} frames) — copying v2 features...")

    for key in ['diff_images','prnu',
                'labels','frame_indices','clip_names','manip_types']:
        src_grp.copy(key, dst_grp)

    dst_grp.create_dataset('temporal_std', (n, 224, 224, 3), dtype='float32',
                           chunks=(4, 224, 224, 3),
                           compression='gzip', compression_opts=4)
    dst_grp.create_dataset('fdi', (n, 42), dtype='float32',
                           chunks=(32, 42),
                           compression='gzip', compression_opts=4)
    dst_grp.create_dataset('dct', (n, 48), dtype='float32',
                           chunks=(32, 48),
                           compression='gzip', compression_opts=4)

    clip_names = [c.decode() if isinstance(c,bytes) else c
                  for c in src_grp['clip_names'][:]]
    labels     = src_grp['labels'][:].astype(int)
    data_root  = Path(data_root)

    # Group rows by clip
    clip_rows = defaultdict(list)
    for row, (cn, lb) in enumerate(zip(clip_names, labels)):
        clip_rows[cn].append((row, lb))

    errors = 0
    print(f"  Computing temporal_std, FDI, DCT for {len(clip_rows)} clips...")

    for clip_name, rows in tqdm(clip_rows.items(),
                                 desc=f"  {split_name}", ncols=72):
        label = rows[0][1]
        sub   = "real" if label == 0 else "fake"
        aligned_dir = data_root / sub / clip_name / "frames_aligned"

        if aligned_dir.exists():
            aligned_paths = sorted(aligned_dir.glob("*.jpg"))
        else:
            aligned_paths = []
            errors += 1

        # Compute once per clip
        temp_std = extract_temporal_consistency(aligned_paths, n_frames=20)
        fdi_feat = extract_fdi_features(aligned_paths, n_frames=20)
        dct_feat = extract_dct_features(aligned_paths, n_frames=5)

        # Write to all rows of this clip
        for row, lb in rows:
            dst_grp['temporal_std'][row] = temp_std
            dst_grp['fdi'][row]          = fdi_feat
            dst_grp['dct'][row]          = dct_feat

    print(f"  {split_name}: done, {errors} clips with missing frames")
    return errors


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot', action='store_true')
    args = p.parse_args()

    mode      = "pilot" if args.pilot else "full"
    src_cache = Path(CACHE_DIR) / f"features_v2_full_seed42.h5"
    dst_cache = Path(CACHE_DIR) / f"features_v3_{mode}_seed42.h5"

    if not src_cache.exists():
        print(f"ERROR: v2 cache not found: {src_cache}")
        sys.exit(1)

    if dst_cache.exists():
        print(f"Overwriting existing v3 cache: {dst_cache}")
        dst_cache.unlink()

    print("="*60)
    print(f"  Feature Precomputation v3 — {mode.upper()}")
    print("="*60)
    print(f"  Source : {src_cache}")
    print(f"  Output : {dst_cache}")
    print(f"  Adding : temporal_std + fdi + dct")

    t0 = time.time()
    errors = 0

    with h5py.File(str(src_cache), 'r') as src, \
         h5py.File(str(dst_cache), 'w') as dst:
        for split in ['train','val','test']:
            if split not in src: continue
            dst_grp = dst.create_group(split)
            errors += process_split(src[split], dst_grp, DATA_ROOT_V2, split)

    elapsed = time.time() - t0
    size_gb = dst_cache.stat().st_size / 1e9

    print(f"\n{'='*60}")
    print(f"  Time       : {elapsed/60:.1f} min")
    print(f"  Cache size : {size_gb:.1f} GB")
    print(f"  Errors     : {errors}")

    print(f"\nVerification:")
    with h5py.File(str(dst_cache), 'r') as f:
        for split in ['train','val','test']:
            if split not in f: continue
            n  = f[split]['labels'].shape[0]
            ts = f[split]['temporal_std'][:4]
            fd = f[split]['fdi'][:10]
            dc = f[split]['dct'][:10]
            print(f"  {split}: n={n}")
            print(f"    temporal_std: mean={ts.mean():.4f}  nan={np.isnan(ts).any()}")
            print(f"    fdi:          mean={fd.mean():.4f}  nan={np.isnan(fd).any()}")
            print(f"    dct:          mean={dc.mean():.4f}  nan={np.isnan(dc).any()}")

    print(f"\n  Next: python main_v3.py {'--pilot' if args.pilot else ''}")
    print("="*60)


if __name__ == "__main__":
    main()