"""
data/dataset_v3.py — v3 dataset with compression augmentation

Key additions vs v2:
  - Loads temporal_std, fdi, dct from v3 cache
  - Concatenates prnu + dct into combined Branch C input
  - Compression augmentation: random JPEG re-compression during training
    Forces model to be robust to different compression levels
    This directly addresses cross-dataset generalization
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import h5py
from PIL import Image
import torchvision.transforms as T
import io

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]


def _jpeg_compress(img_pil, quality):
    """Re-compress PIL image at given JPEG quality and return PIL image."""
    buf = io.BytesIO()
    img_pil.save(buf, format='JPEG', quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


class JPEGAugmentation:
    """Random JPEG re-compression augmentation."""
    def __init__(self, p=0.5, quality_range=(50, 90)):
        self.p      = p
        self.q_low  = quality_range[0]
        self.q_high = quality_range[1]

    def __call__(self, img):
        if torch.rand(1).item() < self.p:
            q = int(torch.randint(self.q_low, self.q_high+1, (1,)).item())
            img = _jpeg_compress(img, q)
        return img


_train_tf = T.Compose([
    JPEGAugmentation(p=0.5, quality_range=(50, 90)),  # compression augmentation
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

# Manipulation type → integer index (for multi-task loss)
TYPE_TO_IDX = {
    'real': 0, 'Deepfakes': 1, 'Face2Face': 2,
    'FaceShifter': 3, 'FaceSwap': 4,
    'NeuralTextures': 5, 'DeepFakeDetection': 6,
}


def _load_aligned(clip_path: Path, frame_idx: int, tf):
    jpg = clip_path / "frames_aligned" / f"{frame_idx:03d}.jpg"
    if not jpg.exists():
        jpgs = sorted((clip_path/"frames_aligned").glob("*.jpg"))
        if not jpgs: return torch.zeros(3, 224, 224)
        jpg = jpgs[min(frame_idx, len(jpgs)-1)]
    try:
        return tf(Image.open(jpg).convert("RGB"))
    except Exception:
        return torch.zeros(3, 224, 224)


class HybridDatasetV3(Dataset):
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

    def _clip_path(self, name, label):
        return self.data_root / ("real" if label==0 else "fake") / name

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r')

    def __len__(self): return self._len

    def __getitem__(self, idx):
        self._open()
        grp   = self._file[self.split]
        clip  = self._clips[idx]
        label = self._labels[idx]
        fi    = int(self._frames[idx])
        mt    = self._types[idx]

        # Cached features
        di  = torch.from_numpy(
            grp['diff_images'][idx].astype(np.float32).transpose(2,0,1))
        ts  = torch.from_numpy(
            grp['temporal_std'][idx].astype(np.float32).transpose(2,0,1))
        fdi = torch.from_numpy(grp['fdi'][idx].astype(np.float32))

        # Branch C: concatenate PRNU + DCT
        prnu = grp['prnu'][idx].astype(np.float32)
        dct  = grp['dct'][idx].astype(np.float32)
        prnu_dct = torch.from_numpy(np.concatenate([prnu, dct]))

        # Raw RGB from disk with compression augmentation
        raw_rgb = _load_aligned(self._clip_path(clip, label), fi, self.tf)

        # Type index for multi-task loss
        type_idx = TYPE_TO_IDX.get(mt, 0)

        return {
            'raw_rgb':      raw_rgb,
            'diff_image':   di,
            'prnu_dct':     prnu_dct,     # 78-dim: 30 prnu + 48 dct
            'temporal_std': ts,
            'fdi':          fdi,
            'label':        torch.tensor(label, dtype=torch.long),
            'type_idx':     torch.tensor(type_idx, dtype=torch.long),
            'clip_name':    clip,
            'manip_type':   mt,
            'frame_idx':    fi,
        }

    def __del__(self):
        if self._file:
            try: self._file.close()
            except: pass


def build_v3_dataloaders(cache_path, data_root,
                          batch_size=12, num_workers=2):
    tr = HybridDatasetV3(cache_path, data_root, 'train', augment=True)
    va = HybridDatasetV3(cache_path, data_root, 'val',   augment=False)
    te = HybridDatasetV3(cache_path, data_root, 'test',  augment=False)
    print(f"  Train={len(tr)}  Val={len(va)}  Test={len(te)}")
    kw = dict(num_workers=num_workers, pin_memory=True,
              persistent_workers=(num_workers>0))
    return (DataLoader(tr, batch_size=batch_size, shuffle=True,  **kw),
            DataLoader(va, batch_size=batch_size, shuffle=False, **kw),
            DataLoader(te, batch_size=batch_size, shuffle=False, **kw))