"""
check_celebdf_official.py
Inspects the official Celeb-DF v2 dataset structure.
Run: python check_celebdf_official.py
"""
import os, sys
from pathlib import Path
import cv2

ROOT = Path("D:/DO NOT TOUCH/CelebDF_v2")

print("="*60)
print("  Official Celeb-DF v2 — Structure Inspection")
print("="*60)
print(f"  Root: {ROOT}\n")

if not ROOT.exists():
    print("ERROR: Path not found"); sys.exit(1)

# Top-level contents
print("TOP-LEVEL CONTENTS:")
for item in sorted(ROOT.iterdir()):
    if item.is_dir():
        children = list(item.iterdir())
        print(f"  [{item.name}/]  ({len(children)} items)")
        for c in sorted(children)[:5]:
            if c.is_dir():
                vids = list(c.glob("*.mp4")) + list(c.glob("*.avi"))
                print(f"    [{c.name}/]  {len(vids)} videos")
            else:
                print(f"    {c.name}  ({c.stat().st_size/1024:.1f} KB)")
    else:
        print(f"  {item.name}  ({item.stat().st_size/1024:.1f} KB)")

# Find all video files and the list file
print("\nLOOKING FOR LIST FILES (train/test splits):")
for f in ROOT.rglob("*.txt"):
    lines = open(f).readlines()
    print(f"  {f.relative_to(ROOT)}: {len(lines)} lines")
    print(f"    First 5: {[l.strip() for l in lines[:5]]}")

print("\nVIDEO FILE SAMPLE:")
all_vids = list(ROOT.rglob("*.mp4"))
print(f"  Total .mp4 files found: {len(all_vids)}")
if all_vids:
    for v in all_vids[:5]:
        cap = cv2.VideoCapture(str(v))
        fps   = cap.get(cv2.CAP_PROP_FPS)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        print(f"  {v.relative_to(ROOT)}: fps={fps:.0f} frames={total} "
              f"duration={total/max(fps,1):.1f}s")

print("\nFOLDER BREAKDOWN:")
for item in sorted(ROOT.iterdir()):
    if item.is_dir():
        vids = list(item.rglob("*.mp4"))
        print(f"  {item.name}/: {len(vids)} videos total")
print("="*60)