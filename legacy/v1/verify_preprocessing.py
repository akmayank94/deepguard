"""
verify_preprocessing.py
Run after: python scripts/preprocess_final.py --test_mode
"""
from pathlib import Path
import numpy as np
import json

out = Path('D:/DO NOT TOUCH/deepfake_microexpression/data/FF++C23_preprocessed')

if not out.exists():
    print(f"ERROR: Output folder not found: {out}")
    print("Run: python scripts/preprocess_final.py --test_mode")
    exit(1)

print("="*60)
print("  Preprocessing Verification")
print("="*60)

all_ok = True
for label in ['real', 'fake']:
    label_dir = out / label
    if not label_dir.exists():
        print(f"  WARNING: {label_dir} not found")
        continue
    clips = sorted(label_dir.iterdir())[:3]
    print(f"\n  [{label}] — checking {len(clips)} clips:")
    for c in clips:
        if not c.is_dir(): continue
        aligned = sorted((c/'frames_aligned').glob('*.jpg')) if (c/'frames_aligned').exists() else []
        raw     = sorted((c/'frames_raw').glob('*.png'))     if (c/'frames_raw').exists() else []
        rppg_path = c / 'rppg.npy'
        meta_path = c / 'meta.json'

        issues = []
        if len(aligned) != 20:
            issues.append(f"aligned={len(aligned)} (need 20)")
        if len(raw) != 64:
            issues.append(f"raw={len(raw)} (need 64)")
        if not rppg_path.exists():
            issues.append("rppg.npy missing")
        if not meta_path.exists():
            issues.append("meta.json missing")

        rppg_ok = False
        rppg_std = 0.0
        if rppg_path.exists():
            rppg = np.load(str(rppg_path))
            rppg_std = float(rppg[:, 1].std())  # green channel std
            if rppg.shape != (64, 3):
                issues.append(f"rppg shape={rppg.shape} (need (64,3))")
            if rppg_std < 0.1:
                issues.append(f"rppg green std={rppg_std:.4f} (too flat, signal may be lost)")
            else:
                rppg_ok = True

        meta = {}
        if meta_path.exists():
            meta = json.load(open(str(meta_path)))

        status = "PASS" if not issues else "FAIL"
        if issues: all_ok = False

        print(f"    {c.name}:")
        print(f"      aligned={len(aligned)}  raw={len(raw)}  "
              f"rppg_shape={rppg.shape if rppg_path.exists() else 'missing'}  "
              f"green_std={rppg_std:.4f}  [{status}]")
        print(f"      fps={meta.get('fps','?')}  "
              f"duration_rppg={meta.get('duration_rppg_seconds','?'):.1f}s  "
              f"effective_fps={meta.get('effective_fps_rppg','?')}")
        if issues:
            for iss in issues:
                print(f"      !! {iss}")

        # Show rPPG signal variation (key diagnostic)
        if rppg_path.exists():
            rppg = np.load(str(rppg_path))
            print(f"      rPPG R mean={rppg[:,0].mean():.1f} std={rppg[:,0].std():.3f}")
            print(f"      rPPG G mean={rppg[:,1].mean():.1f} std={rppg[:,1].std():.3f}")
            print(f"      rPPG B mean={rppg[:,2].mean():.1f} std={rppg[:,2].std():.3f}")

# Check one aligned JPG and one raw PNG visually
print(f"\n  File size check (aligned JPG vs raw PNG):")
for label in ['real']:
    clips = sorted((out/label).iterdir())[:1]
    for c in clips:
        j = sorted((c/'frames_aligned').glob('*.jpg'))[0] if (c/'frames_aligned').exists() else None
        p = sorted((c/'frames_raw').glob('*.png'))[0]     if (c/'frames_raw').exists() else None
        if j: print(f"    Aligned JPG: {j.stat().st_size/1024:.1f} KB  ({j.name})")
        if p: print(f"    Raw    PNG: {p.stat().st_size/1024:.1f} KB  ({p.name})")
        print(f"    (PNG should be larger than JPG — lossless compression)")

print(f"\n{'='*60}")
print(f"  Overall: {'ALL PASS — ready for full overnight run' if all_ok else 'ISSUES FOUND — fix before full run'}")
print(f"  Command: python scripts/preprocess_final.py")
print(f"{'='*60}")