"""
diag2_clip_overlap.py

FF++ clip names encode source video IDs: "Deepfakes_274_412" means
source video 274 was swapped with source video 412.

If video 274's real clip is in training but "Deepfakes_274_412" is in
validation, the model might learn source-video-specific features that
transfer between train and val — giving falsely high val AUC, OR
conversely, the model might learn features specific to source 274's
camera that DIFFER between real and fake versions, hurting val AUC.

This diagnostic checks how much source video overlap exists between
train and val splits.
"""
import os, sys, re
import numpy as np
from pathlib import Path
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

import h5py
from collections import defaultdict

cache_dir  = os.environ.get("CACHE_DIR", "cache")
cache_path = Path(cache_dir) / "features_full_seed42.h5"

print("="*60)
print("  Source Video Overlap Analysis")
print("="*60)

def extract_video_ids(clip_name):
    """Extract source video IDs from clip name."""
    nums = re.findall(r'\d+', clip_name)
    return set(nums)

with h5py.File(cache_path, 'r') as f:
    for split in ('train', 'val', 'test'):
        clips  = [c.decode() if isinstance(c,bytes) else c
                  for c in f[split]['clip_names'][:]]
        labels = f[split]['labels'][:]
        unique_clips = list(set(clips))
        print(f"\n  {split}: {len(unique_clips)} unique clips "
              f"({int((labels==0).sum())} real frames, "
              f"{int((labels==1).sum())} fake frames)")

    # Get unique clip names per split
    tr_clips = set(c.decode() if isinstance(c,bytes) else c
                   for c in f['train']['clip_names'][:])
    va_clips = set(c.decode() if isinstance(c,bytes) else c
                   for c in f['val']['clip_names'][:])
    te_clips = set(c.decode() if isinstance(c,bytes) else c
                   for c in f['test']['clip_names'][:])

# Extract source video IDs
def get_video_ids(clip_set):
    ids = set()
    for clip in clip_set:
        ids.update(extract_video_ids(clip))
    return ids

tr_ids = get_video_ids(tr_clips)
va_ids = get_video_ids(va_clips)
te_ids = get_video_ids(te_clips)

tr_va_overlap = tr_ids & va_ids
tr_te_overlap = tr_ids & te_ids

print(f"\n  Unique source video IDs in train : {len(tr_ids)}")
print(f"  Unique source video IDs in val   : {len(va_ids)}")
print(f"  Unique source video IDs in test  : {len(te_ids)}")
print(f"\n  Train-Val source video overlap   : {len(tr_va_overlap)} IDs")
print(f"  Train-Test source video overlap  : {len(tr_te_overlap)} IDs")

if len(tr_va_overlap) > 0:
    pct = len(tr_va_overlap) / len(va_ids) * 100
    print(f"\n  {pct:.1f}% of val video IDs also appear in training")
    if pct > 50:
        print("  HIGH OVERLAP — model might generalize via source video identity")
        print("  This can be good (real signal) or bad (spurious correlation)")
    else:
        print("  LOW OVERLAP — val tests genuine generalization to new videos")

# Show example clips
print(f"\n  Example train clips: {list(tr_clips)[:5]}")
print(f"  Example val   clips: {list(va_clips)[:5]}")

# Check if any exact clip names appear in both splits
exact_overlap = tr_clips & va_clips
print(f"\n  Exact clip name overlap: {len(exact_overlap)} (should be 0)")
if exact_overlap:
    print(f"  WARNING: {list(exact_overlap)[:3]}")
else:
    print("  PASS — no clip appears in both train and val")