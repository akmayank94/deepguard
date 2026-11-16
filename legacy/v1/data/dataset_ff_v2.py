"""
data/dataset_ff_v2.py — updated with DeepFakeDetection support

DeepFakeDetection clips split by actor pair to prevent leakage.
All other types split by source video ID as before.
Balance: 1:1 real/fake via undersampling per type.
"""

import re
import random
from pathlib import Path
from typing import Tuple, List, Dict

LABEL_REAL = 0
LABEL_FAKE = 1

FAKE_TYPES = [
    "Deepfakes", "Face2Face", "FaceShifter",
    "FaceSwap", "NeuralTextures", "DeepFakeDetection"
]


def _extract_source_id(clip_name: str) -> int:
    nums = re.findall(r'\d+', clip_name)
    return int(nums[0]) if nums else -1


def _extract_actor_pair(clip_name: str) -> str:
    stem  = clip_name.replace("DeepFakeDetection_", "")
    parts = stem.split("__")[0]
    return parts


def _collect_clips(data_root: Path):
    real_dir   = data_root / "real"
    real_clips = sorted([d for d in real_dir.iterdir()
                         if d.is_dir() and (d/"frames_aligned").exists()]) \
                 if real_dir.exists() else []
    fake_by_type = {}
    fake_dir = data_root / "fake"
    if fake_dir.exists():
        for ft in FAKE_TYPES:
            clips = sorted([d for d in fake_dir.iterdir()
                           if d.is_dir() and d.name.startswith(ft+"_")
                           and (d/"frames_aligned").exists()])
            if clips:
                fake_by_type[ft] = clips
    return real_clips, fake_by_type


def build_ff_v2_splits(
    data_root: str,
    train_frac: float = 0.70,
    val_frac:   float = 0.15,
    seed: int   = 42,
    pilot: bool = False,
) -> Tuple[list, list, list]:
    data_root  = Path(data_root)
    real_clips, fake_by_type = _collect_clips(data_root)
    rng = random.Random(seed)

    if pilot:
        real_clips = real_clips[:100]
        fake_by_type = {k: v[:20] for k, v in fake_by_type.items()}

    all_real_ids = sorted(set(_extract_source_id(c.name) for c in real_clips))
    rng.shuffle(all_real_ids)
    n        = len(all_real_ids)
    n_train  = int(n * train_frac)
    n_val    = int(n * val_frac)
    train_ids = set(all_real_ids[:n_train])
    val_ids   = set(all_real_ids[n_train:n_train+n_val])
    test_ids  = set(all_real_ids[n_train+n_val:])

    def _entry(path, label, manip_type):
        return {'path': path, 'label': label,
                'manip_type': manip_type, 'clip_name': path.name}

    def _by_source_id(clips, label, manip_type):
        tr,va,te = [],[],[]
        for c in clips:
            sid = _extract_source_id(c.name)
            e   = _entry(c, label, manip_type)
            if   sid in train_ids: tr.append(e)
            elif sid in val_ids:   va.append(e)
            else:                  te.append(e)
        return tr, va, te

    def _by_actor_pair(clips, label, manip_type):
        pairs   = sorted(set(_extract_actor_pair(c.name) for c in clips))
        rng.shuffle(pairs)
        np_     = len(pairs)
        tr_p    = set(pairs[:int(np_*train_frac)])
        va_p    = set(pairs[int(np_*train_frac):int(np_*(train_frac+val_frac))])
        tr,va,te = [],[],[]
        for c in clips:
            pair = _extract_actor_pair(c.name)
            e    = _entry(c, label, manip_type)
            if   pair in tr_p: tr.append(e)
            elif pair in va_p: va.append(e)
            else:              te.append(e)
        return tr, va, te

    real_tr, real_va, real_te = _by_source_id(real_clips, LABEL_REAL, "real")

    n_types        = len(fake_by_type)
    per_type_train = len(real_tr) // n_types
    per_type_val   = len(real_va) // n_types
    per_type_test  = len(real_te) // n_types

    train_all = list(real_tr)
    val_all   = list(real_va)
    test_all  = list(real_te)

    for ft, clips in fake_by_type.items():
        if ft == "DeepFakeDetection":
            tr, va, te = _by_actor_pair(clips, LABEL_FAKE, ft)
        else:
            tr, va, te = _by_source_id(clips, LABEL_FAKE, ft)
        rng.shuffle(tr); rng.shuffle(va); rng.shuffle(te)
        train_all.extend(tr[:per_type_train])
        val_all.extend(va[:per_type_val])
        test_all.extend(te[:per_type_test])

    rn = sum(1 for e in train_all if e['label']==0)
    fn = sum(1 for e in train_all if e['label']==1)
    print(f"  Split sizes — train:{len(train_all)}  val:{len(val_all)}  test:{len(test_all)}")
    print(f"  Train balance — real:{rn}  fake:{fn}  ratio:{fn/max(rn,1):.2f}")
    print(f"  Fake types included: {list(fake_by_type.keys())}")
    return train_all, val_all, test_all