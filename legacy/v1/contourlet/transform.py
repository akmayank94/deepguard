"""
contourlet/transform.py — unchanged from v1 (pipeline was correct).
"""
import numpy as np
from typing import Optional
from .pyramidal import LaplacianPyramid
from .directional import DirectionalFilterBank

class ContourletTransform:
    def __init__(self, num_levels=2, directions_per_level=None):
        if directions_per_level is None:
            directions_per_level = [4, 8]
        if len(directions_per_level) != num_levels:
            raise ValueError("len(directions_per_level) must equal num_levels")
        for k in directions_per_level:
            if k not in (4, 8):
                raise ValueError(f"directions must be 4 or 8, got {k}")
        self.num_levels = num_levels
        self.directions_per_level = directions_per_level
        self.lp = LaplacianPyramid(num_levels=num_levels)
        self.dfbs = [DirectionalFilterBank(num_directions=k) for k in directions_per_level]

    def decompose_image(self, image):
        if image.ndim == 2:
            image = np.stack([image, image, image], axis=-1)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError(f"Expected (H,W,3), got {image.shape}")
        H, W = image.shape[:2]
        img_f64 = image.astype(np.float64)
        if img_f64.max() <= 1.0 + 1e-6:
            img_f64 *= 255.0
        channels = {'R': img_f64[:,:,0], 'G': img_f64[:,:,1], 'B': img_f64[:,:,2]}
        lp_results = {ch: self.lp.decompose(ch_img) for ch, ch_img in channels.items()}
        subbands = {ch: [] for ch in ('R','G','B')}
        for ch in ('R','G','B'):
            for level_idx, dfb in enumerate(self.dfbs):
                bp = lp_results[ch]['bandpass'][level_idx]
                subbands[ch].append(dfb.decompose(bp))
        coarse  = {ch: lp_results[ch]['coarse']   for ch in ('R','G','B')}
        bandpass= {ch: lp_results[ch]['bandpass']  for ch in ('R','G','B')}
        meta = {
            'image_shape': (H, W),
            'num_levels': self.num_levels,
            'directions_per_level': self.directions_per_level,
            'subband_shapes': {
                ch: [[s.shape for s in subbands[ch][l]] for l in range(self.num_levels)]
                for ch in ('R','G','B')
            },
        }
        return {'subbands': subbands, 'coarse': coarse, 'bandpass': bandpass, 'meta': meta}

    def get_level1_subbands(self, result):
        return {ch: result['subbands'][ch][0] for ch in ('R','G','B')}

    def get_dominant_direction_maps(self, result, level=0):
        dfb = self.dfbs[level]
        return {ch: dfb.get_dominant_direction_map(result['subbands'][ch][level])
                for ch in ('R','G','B')}

    def print_summary(self, result):
        meta = result['meta']
        print(f"ContourletTransform: shape={meta['image_shape']}, "
              f"levels={meta['num_levels']}, dirs={meta['directions_per_level']}")
        for ch in ('R','G','B'):
            for l in range(meta['num_levels']):
                shapes = meta['subband_shapes'][ch][l]
                print(f"  [{ch}] level {l+1}: {[str(s) for s in shapes]}")
            print(f"  [{ch}] coarse: {result['coarse'][ch].shape}")

def build_transform(config=None):
    if config is None: config = {}
    return ContourletTransform(
        num_levels=config.get('levels', 2),
        directions_per_level=config.get('directions', [4, 8]),
    )