"""
features/extractor.py — final

No per-image normalization. Raw features concatenated and saved to HDF5.
Normalization happens at training time via the HandcraftedBranch's
input BatchNorm1d — which normalizes across the batch using running
statistics learned during training. This is the correct approach.
"""

import numpy as np
from typing import Optional
from contourlet import ContourletTransform
from features.energy import InterChannelEnergyExtractor
from features.dccm   import DCCMExtractor


class ForensicFeatureExtractor:
    def __init__(self, ct: Optional[ContourletTransform] = None):
        self.ct = ct  # kept for API compat
        self.energy_extractor = InterChannelEnergyExtractor()
        self.dccm_extractor   = DCCMExtractor()

    def extract(self, image: np.ndarray) -> dict:
        e = self.energy_extractor.extract(image)
        d = self.dccm_extractor.extract(image)

        # Simple concatenation — BatchNorm in model handles normalization
        handcrafted = np.concatenate([
            e['feature_vector'].astype(np.float32),
            d['feature_vector'].astype(np.float32),
        ])

        return {
            'handcrafted_vector': handcrafted,
            'diff_image':         e['diff_image'],
            'energy_vector':      e['feature_vector'].astype(np.float32),
            'dccm_vector':        d['feature_vector'].astype(np.float32),
            'energy_result':      e,
            'dccm_result':        d,
        }

    def feature_dims(self) -> dict:
        e = self.energy_extractor.feature_dim()
        d = self.dccm_extractor.feature_dim()
        return {'energy': e, 'dccm': d, 'handcrafted': e + d}


def extract_features_for_clip(clip_path, extractor, frames_per_clip=20):
    from PIL import Image as PILImage
    from pathlib import Path
    results = []
    for jpg in sorted(Path(clip_path).glob("*.jpg")):
        try:
            img = np.array(PILImage.open(jpg).convert("RGB"))
            r   = extractor.extract(img)
            results.append({
                'handcrafted_vector': r['handcrafted_vector'],
                'diff_image':         r['diff_image'],
                'energy_vector':      r['energy_vector'],
                'dccm_vector':        r['dccm_vector'],
                'frame_path':         str(jpg),
            })
        except Exception as ex:
            print(f"  WARNING: {jpg}: {ex}")
    return results