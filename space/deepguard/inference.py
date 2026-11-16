"""
deepguard/inference.py — model loading and the batched forward pass.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .models.detector_v2 import build_detector_v2

IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)

BRANCH_NAMES = ["Raw spatial", "Spectral diff", "CT handcrafted", "PRNU noise", "rPPG temporal"]


def load_checkpoint(path: str | Path, device: str):
    """
    Loads the training checkpoint saved by train/trainer_v2.py:
      {'model': state_dict, 'config': {...}, 'epoch': int, 'val_auc': float, ...}
    Loading is strict: any missing/unexpected weight raises, so a silent
    architecture mismatch can never reach users.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Upload model.pt to the Space.")
    # weights_only=False: this is our own training checkpoint (contains a config dict).
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
    config = ckpt.get("config", {}) if isinstance(ckpt, dict) else {}
    model = build_detector_v2(config, pretrained_raw=False)
    model.load_state_dict(state, strict=True)
    model.eval().to(device)
    meta = {
        "epoch": ckpt.get("epoch") if isinstance(ckpt, dict) else None,
        "val_auc": ckpt.get("val_auc") if isinstance(ckpt, dict) else None,
        "n_params": sum(p.numel() for p in model.parameters()),
    }
    return model, meta


def build_tensors(clip):
    """Numpy clip inputs -> CPU tensors in the exact format of dataset_v2."""
    rgb = torch.from_numpy(np.stack(clip.aligned_rgb)).permute(0, 3, 1, 2).float() / 255.0
    raw = (rgb - IMAGENET_MEAN) / IMAGENET_STD                     # T.ToTensor + T.Normalize
    raw_flip = torch.flip(raw, dims=[3])                            # horizontal flip (TTA)
    diff = torch.from_numpy(clip.diff_images.transpose(0, 3, 1, 2)).float()
    hc = torch.from_numpy(clip.handcrafted).float()
    n = raw.shape[0]
    prnu = torch.from_numpy(clip.prnu).float().unsqueeze(0).repeat(n, 1)
    rppg = torch.from_numpy(clip.rppg).float().unsqueeze(0).repeat(n, 1)
    return raw, raw_flip, diff, hc, prnu, rppg


@torch.no_grad()
def forward_clip(model, tensors, use_tta: bool):
    device = next(model.parameters()).device
    raw, raw_flip, diff, hc, prnu, rppg = [t.to(device) for t in tensors]
    logits, gates = model(raw, diff, hc, prnu, rppg, return_gates=True)
    probs = torch.sigmoid(logits)
    if use_tta:
        logits_f, gates_f = model(raw_flip, diff, hc, prnu, rppg, return_gates=True)
        probs = (probs + torch.sigmoid(logits_f)) / 2.0
        gates = (gates + gates_f) / 2.0
    frame_probs = probs.float().cpu().numpy().tolist()
    return {
        "frame_probs": frame_probs,
        "fake_probability": float(np.mean(frame_probs)),
        "gate_weights": gates.float().mean(dim=0).cpu().numpy().tolist(),
    }
