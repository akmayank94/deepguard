"""
data/dataset_hybrid.py

All 20 frames per clip for training with heavy augmentation.
Heavy augmentation prevents clip memorization without reducing data size.
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
from PIL import Image
import torchvision.transforms as T

CLIP_LO = -500.0
CLIP_HI =  500.0
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

_train_tf = T.Compose([
    T.RandomHorizontalFlip(p=0.5),
    T.RandomRotation(degrees=10),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1, hue=0.05),
    T.RandomGrayscale(p=0.05),
    T.GaussianBlur(kernel_size=3, sigma=(0.1, 1.0)),
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    T.RandomErasing(p=0.2, scale=(0.02, 0.1)),
])

_eval_tf = T.Compose([
    T.ToTensor(),
    T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
])


def _load_frame(clip_path: Path, frame_idx: int, tf) -> torch.Tensor:
    jpg = clip_path / f"{frame_idx:03d}.jpg"
    if not jpg.exists():
        jpgs = sorted(clip_path.glob("*.jpg"))
        if not jpgs:
            return torch.zeros(3, 224, 224)
        jpg = jpgs[0]
    try:
        return tf(Image.open(jpg).convert("RGB"))
    except Exception:
        return torch.zeros(3, 224, 224)


class HybridDataset(Dataset):
    def __init__(self, cache_path, data_root, split='train', augment=True):
        self.cache_path = str(cache_path)
        self.data_root  = Path(data_root)
        self.split      = split
        self.tf         = _train_tf if augment else _eval_tf
        self._file      = None

        with h5py.File(self.cache_path, 'r') as f:
            self._len    = f[split]['labels'].shape[0]
            self._clips  = [c.decode() if isinstance(c,bytes) else c
                            for c in f[split]['clip_names'][:]]
            self._labels = f[split]['labels'][:].astype(int)
            self._types  = [t.decode() if isinstance(t,bytes) else t
                            for t in f[split]['manip_types'][:]]
            self._frames = f[split]['frame_indices'][:].astype(int)

    def _clip_path(self, name, label):
        sub = "real" if label == 0 else "fake"
        return self.data_root / sub / name

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r')

    def __len__(self):
        return self._len

    def __getitem__(self, idx):
        self._open()
        grp   = self._file[self.split]
        clip  = self._clips[idx]
        label = self._labels[idx]
        fi    = int(self._frames[idx])

        hc = torch.from_numpy(
            np.clip(grp['handcrafted'][idx], CLIP_LO, CLIP_HI).astype(np.float32))
        di = torch.from_numpy(
            grp['diff_images'][idx].astype(np.float32).transpose(2,0,1))
        rr = _load_frame(self._clip_path(clip, label), fi, self.tf)

        return {
            'raw_rgb':     rr,
            'diff_image':  di,
            'handcrafted': hc,
            'label':       torch.tensor(label, dtype=torch.long),
            'clip_name':   clip,
            'manip_type':  self._types[idx],
            'frame_idx':   fi,
        }

    def __del__(self):
        if self._file:
            try: self._file.close()
            except: pass


def build_hybrid_dataloaders(cache_path, data_root, batch_size=32, num_workers=2):
    tr = HybridDataset(cache_path, data_root, 'train', augment=True)
    va = HybridDataset(cache_path, data_root, 'val',   augment=False)
    te = HybridDataset(cache_path, data_root, 'test',  augment=False)
    print(f"  Train={len(tr)}  Val={len(va)}  Test={len(te)}")
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers>0))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(va, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(te, batch_size=batch_size, shuffle=False, **kw))