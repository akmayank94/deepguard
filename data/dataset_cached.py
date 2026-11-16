"""
data/dataset_cached.py — final

Key fix: ClipSampledDataset for training uses ONE random frame per clip
per epoch. This prevents the model from memorizing specific frames of
specific clips and forces it to learn generalizable spectral patterns.

For validation and test: all frames used (no subsampling).
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
from collections import defaultdict
import h5py

CLIP_FEATURE = -500.0
CLIP_CEILING =  500.0


class CachedFeatureDataset(Dataset):
    """Standard dataset — all frames. Used for val and test."""

    def __init__(self, cache_path, split, augment_diff_image=False):
        self.cache_path         = str(cache_path)
        self.split              = split
        self.augment_diff_image = augment_diff_image
        self._file              = None
        with h5py.File(self.cache_path, 'r') as f:
            if split not in f:
                raise ValueError(f"Split '{split}' not in cache")
            self._length = f[split]['labels'].shape[0]

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r')

    def __len__(self):
        return self._length

    def __getitem__(self, idx):
        self._open()
        grp = self._file[self.split]

        # Clip handcrafted features to prevent BatchNorm variance explosion
        handcrafted = torch.from_numpy(
            np.clip(grp['handcrafted'][idx],
                    CLIP_FEATURE, CLIP_CEILING).astype(np.float32))

        diff_img = grp['diff_images'][idx].astype(np.float32)
        if self.augment_diff_image and np.random.random() < 0.5:
            diff_img = diff_img[:, ::-1, :].copy()
        diff_img = torch.from_numpy(diff_img.transpose(2, 0, 1))

        label      = int(grp['labels'][idx])
        frame_idx  = int(grp['frame_indices'][idx])
        clip_name  = grp['clip_names'][idx]
        manip_type = grp['manip_types'][idx]
        if isinstance(clip_name,  bytes): clip_name  = clip_name.decode()
        if isinstance(manip_type, bytes): manip_type = manip_type.decode()

        return {
            'handcrafted': handcrafted,
            'diff_image':  diff_img,
            'label':       torch.tensor(label, dtype=torch.long),
            'frame_idx':   frame_idx,
            'clip_name':   clip_name,
            'manip_type':  manip_type,
        }

    def label_distribution(self):
        with h5py.File(self.cache_path, 'r') as f:
            labels = f[self.split]['labels'][:]
        counts = defaultdict(int)
        for lb in labels:
            counts[int(lb)] += 1
        return dict(counts)

    def __del__(self):
        if self._file is not None:
            try: self._file.close()
            except: pass


class ClipSampledDataset(Dataset):
    """
    Training dataset that samples ONE random frame per clip per epoch.

    Why this matters:
        With 20 frames per clip and 1,383 training clips, a standard
        dataset has 27,660 frames. The model quickly memorizes which
        clip-specific frame patterns are real or fake. By sampling one
        frame per clip per epoch, we force the model to see diverse
        frames and learn clip-agnostic spectral patterns.

        This directly addresses the train AUC=0.99 / val AUC=0.44 gap.

    Each call to resample() picks a new random frame per clip.
    The trainer calls resample() at the start of each epoch.
    """

    def __init__(self, cache_path, split='train', augment=True, seed=42):
        self.cache_path = str(cache_path)
        self.split      = split
        self.augment    = augment
        self.rng        = np.random.RandomState(seed)
        self._file      = None

        # Build clip index: clip_name -> list of HDF5 row indices
        print(f"  Building clip index for {split} split...")
        with h5py.File(self.cache_path, 'r') as f:
            n = f[split]['labels'].shape[0]
            clip_names  = [c.decode() if isinstance(c, bytes) else c
                           for c in f[split]['clip_names'][:]]
            labels      = f[split]['labels'][:].astype(int)
            manip_types = [t.decode() if isinstance(t, bytes) else t
                           for t in f[split]['manip_types'][:]]

        self.clip_to_indices = defaultdict(list)
        self.clip_label      = {}
        self.clip_manip      = {}
        for i, clip in enumerate(clip_names):
            self.clip_to_indices[clip].append(i)
            self.clip_label[clip] = labels[i]
            self.clip_manip[clip] = manip_types[i]

        self.clips = sorted(self.clip_to_indices.keys())
        self._sampled_indices = None
        self.resample()  # Initial sample
        print(f"  ClipSampledDataset: {len(self.clips)} clips → "
              f"{len(self._sampled_indices)} frames per epoch")

    def resample(self):
        """Pick one random frame per clip. Call at start of each epoch."""
        sampled = []
        for clip in self.clips:
            indices = self.clip_to_indices[clip]
            chosen  = self.rng.choice(indices)
            sampled.append(chosen)
        self._sampled_indices = sampled

    def _open(self):
        if self._file is None:
            self._file = h5py.File(self.cache_path, 'r')

    def __len__(self):
        return len(self._sampled_indices)

    def __getitem__(self, pos):
        self._open()
        idx = self._sampled_indices[pos]
        grp = self._file[self.split]

        handcrafted = torch.from_numpy(
            np.clip(grp['handcrafted'][idx],
                    CLIP_FEATURE, CLIP_CEILING).astype(np.float32))

        diff_img = grp['diff_images'][idx].astype(np.float32)
        if self.augment and np.random.random() < 0.5:
            diff_img = diff_img[:, ::-1, :].copy()
        if self.augment and np.random.random() < 0.3:
            # Random brightness jitter on diff image
            factor = np.random.uniform(0.85, 1.15)
            diff_img = np.clip(diff_img * factor, 0, 1)
        diff_img = torch.from_numpy(diff_img.transpose(2, 0, 1))

        clip_name  = self.clips[pos]
        label      = self.clip_label[clip_name]
        manip_type = self.clip_manip[clip_name]
        frame_idx  = int(grp['frame_indices'][idx])

        return {
            'handcrafted': handcrafted,
            'diff_image':  diff_img,
            'label':       torch.tensor(label, dtype=torch.long),
            'frame_idx':   frame_idx,
            'clip_name':   clip_name,
            'manip_type':  manip_type,
        }

    def __del__(self):
        if self._file is not None:
            try: self._file.close()
            except: pass


def build_cached_dataloaders(cache_path, batch_size=32, num_workers=0):
    """
    Build DataLoaders. Training uses ClipSampledDataset (1 frame/clip/epoch).
    Val and test use full CachedFeatureDataset.
    """
    train_ds = ClipSampledDataset(cache_path, 'train', augment=True)
    val_ds   = CachedFeatureDataset(cache_path, 'val',   augment_diff_image=False)
    test_ds  = CachedFeatureDataset(cache_path, 'test',  augment_diff_image=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size,
                              shuffle=True, num_workers=num_workers,
                              pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size,
                              shuffle=False, num_workers=num_workers,
                              pin_memory=True)
    return train_loader, val_loader, test_loader