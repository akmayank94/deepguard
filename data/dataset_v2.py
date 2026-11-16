"""
data/dataset_v2.py

Dataset for five-branch model:
  1. raw_rgb      — aligned 224x224 JPG loaded from disk → EfficientNet
  2. diff_image   — cached inter-channel energy map       → Spectral EfficientNet
  3. handcrafted  — cached CT energy + DCCM              → MLP branch
  4. prnu         — cached noise residual features        → PRNU MLP
  5. rppg         — cached heartbeat frequency features   → rPPG MLP

Training: all 20 frames per clip, heavy augmentation on raw_rgb only.
Val/test: all frames, no augmentation.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
from PIL import Image
import torchvision.transforms as T

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_train_tf = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=8),
    T.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.08),
    T.RandomGrayscale(p=0.03),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 0.8)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    T.RandomErasing(p=0.15, scale=(0.02, 0.08)),
])
_eval_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])

CLIP_LO = -500.0
CLIP_HI  =  500.0


def _load_aligned(clip_path: Path, frame_idx: int, tf) -> torch.Tensor:
    jpg = clip_path / "frames_aligned" / f"{frame_idx:03d}.jpg"
    if not jpg.exists():
        jpgs = sorted((clip_path/"frames_aligned").glob("*.jpg"))
        if not jpgs: return torch.zeros(3, 224, 224)
        jpg = jpgs[min(frame_idx, len(jpgs)-1)]
    try:
        return tf(Image.open(jpg).convert("RGB"))
    except Exception:
        return torch.zeros(3, 224, 224)


class HybridDatasetV2(Dataset):
    """Full-frame dataset for five-branch model."""

    def __init__(self, cache_path, data_root, split='train', augment=True):
        self.cache_path = str(cache_path)
        self.data_root  = Path(data_root)
        self.split      = split
        self.tf         = _train_tf if augment else _eval_tf
        self._file      = None

        with h5py.File(self.cache_path, 'r') as f:
            grp          = f[split]
            self._len    = grp['labels'].shape[0]
            self._clips  = [c.decode() if isinstance(c,bytes) else c
                            for c in grp['clip_names'][:]]
            self._labels = grp['labels'][:].astype(int)
            self._types  = [t.decode() if isinstance(t,bytes) else t
                            for t in grp['manip_types'][:]]
            self._frames = grp['frame_indices'][:].astype(int)

    def _clip_path(self, clip_name, label):
        sub = "real" if label == 0 else "fake"
        return self.data_root / sub / clip_name

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r')

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._open()
        grp       = self._file[self.split]
        clip_name = self._clips[idx]
        label     = self._labels[idx]
        fi        = int(self._frames[idx])

        # Cached features (same for all 20 frames of a clip for prnu/rppg)
        hc   = torch.from_numpy(
            np.clip(grp['handcrafted'][idx], CLIP_LO, CLIP_HI).astype(np.float32))
        di   = torch.from_numpy(
            grp['diff_images'][idx].astype(np.float32).transpose(2, 0, 1))
        prnu = torch.from_numpy(grp['prnu'][idx].astype(np.float32))
        rppg = torch.from_numpy(grp['rppg'][idx].astype(np.float32))

        # Raw RGB from disk (augmented for training)
        clip_path = self._clip_path(clip_name, label)
        raw_rgb   = _load_aligned(clip_path, fi, self.tf)

        return {
            'raw_rgb':     raw_rgb,
            'diff_image':  di,
            'handcrafted': hc,
            'prnu':        prnu,
            'rppg':        rppg,
            'label':       torch.tensor(label, dtype=torch.long),
            'clip_name':   clip_name,
            'manip_type':  self._types[idx],
            'frame_idx':   fi,
        }

    def __del__(self):
        if self._file:
            try: self._file.close()
            except: pass


def build_v2_dataloaders(cache_path, data_root,
                          batch_size=24, num_workers=2):
    tr = HybridDatasetV2(cache_path, data_root, 'train', augment=True)
    va = HybridDatasetV2(cache_path, data_root, 'val',   augment=False)
    te = HybridDatasetV2(cache_path, data_root, 'test',  augment=False)
    print(f"  Train={len(tr)}  Val={len(va)}  Test={len(te)}")
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers > 0))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(va, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(te, batch_size=batch_size, shuffle=False, **kw))