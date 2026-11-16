"""verify_celebdf_preprocess.py — run after test mode to confirm quality"""
from pathlib import Path
import numpy as np
from PIL import Image

OUT = Path("D:/DO NOT TOUCH/CelebDF_v2_test_preprocessed")

print("="*50)
print("  CelebDF Preprocessing Verification")
print("="*50)

for label in ["real","fake"]:
    clips = sorted((OUT/label).iterdir()) if (OUT/label).exists() else []
    print(f"\n  {label}: {len(clips)} clips")
    for c in clips[:3]:
        aligned = sorted((c/"frames_aligned").glob("*.jpg"))
        raw     = sorted((c/"frames_raw").glob("*.png"))
        rppg    = np.load(str(c/"rppg.npy")) if (c/"rppg.npy").exists() else None
        img     = np.array(Image.open(aligned[0]).convert("RGB")) if aligned else None
        print(f"    {c.name}:")
        print(f"      aligned={len(aligned)} raw={len(raw)} "
              f"rppg={rppg.shape if rppg is not None else 'missing'}")
        if img is not None:
            print(f"      frame size={img.shape}  "
                  f"mean={img.mean():.1f}  std={img.std():.1f}")
        if rppg is not None:
            print(f"      rppg green std={rppg[:,1].std():.4f}")

print("\n  PASS if: aligned=20, raw=64, rppg=(64,3), frame size=(224,224,3)")
print("="*50)