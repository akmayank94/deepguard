"""
contourlet/pyramidal.py  — unchanged from v1, reconstruction was already correct.
"""
import numpy as np
from .filters import (
    PKVA_LOWPASS_ANALYSIS, PKVA_LOWPASS_SYNTHESIS, conv2d_separable_mirror,
)

def _lowpass_downsample(image):
    h0 = PKVA_LOWPASS_ANALYSIS
    filtered = conv2d_separable_mirror(image, h0, h0)
    return filtered[::2, ::2]

def _upsample_lowpass(coarse, target_shape):
    H, W = target_shape
    Hc, Wc = coarse.shape
    upsampled = np.zeros((Hc * 2, Wc * 2), dtype=np.float64)
    upsampled[::2, ::2] = coarse
    g0 = PKVA_LOWPASS_SYNTHESIS
    prediction = conv2d_separable_mirror(upsampled, g0, g0) * 4.0
    prediction = prediction[:H, :W]
    if prediction.shape[0] < H:
        prediction = np.pad(prediction, ((0, H - prediction.shape[0]), (0, 0)))
    if prediction.shape[1] < W:
        prediction = np.pad(prediction, ((0, 0), (0, W - prediction.shape[1])))
    return prediction

class LaplacianPyramid:
    def __init__(self, num_levels=2):
        if num_levels < 1:
            raise ValueError(f"num_levels must be >= 1, got {num_levels}")
        self.num_levels = num_levels

    def decompose(self, image):
        image = image.astype(np.float64)
        bandpass_layers, shapes = [], []
        current = image
        for _ in range(self.num_levels):
            shapes.append(current.shape)
            coarse = _lowpass_downsample(current)
            prediction = _upsample_lowpass(coarse, current.shape)
            bandpass_layers.append(current - prediction)
            current = coarse
        return {'bandpass': bandpass_layers, 'coarse': current, 'shapes': shapes}

    def reconstruct(self, lp_result):
        current = lp_result['coarse'].astype(np.float64)
        for level in range(self.num_levels - 1, -1, -1):
            prediction = _upsample_lowpass(current, lp_result['shapes'][level])
            current = prediction + lp_result['bandpass'][level]
        return current

    def decompose_multichannel(self, image_rgb):
        assert image_rgb.ndim == 3 and image_rgb.shape[2] == 3
        return {
            'R': self.decompose(image_rgb[:, :, 0]),
            'G': self.decompose(image_rgb[:, :, 1]),
            'B': self.decompose(image_rgb[:, :, 2]),
        }

def verify_reconstruction(image, num_levels=2, tol=1e-4):
    lp = LaplacianPyramid(num_levels=num_levels)
    result = lp.decompose(image)
    recon = lp.reconstruct(result)
    max_err = np.max(np.abs(recon - image))
    assert max_err < tol, f"LP reconstruction error {max_err:.2e} > {tol:.2e}"
    return True

def check_energy_preservation(image, num_levels=2):
    lp = LaplacianPyramid(num_levels=num_levels)
    result = lp.decompose(image)
    input_energy = np.sum(image.astype(np.float64) ** 2)
    subband_energy = sum(np.sum(bp**2) for bp in result['bandpass'])
    subband_energy += np.sum(result['coarse'] ** 2)
    return {
        'input_energy': input_energy,
        'total_subband_energy': subband_energy,
        'ratio': subband_energy / (input_energy + 1e-12),
    }