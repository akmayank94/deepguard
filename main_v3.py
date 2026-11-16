"""
main_v3.py — Complete five-branch detector v3

Branches:
  A: EfficientNet-B3 raw RGB (spatial appearance)
  B: EfficientNet-B0 spectral diff_image (inter-channel frequency)
  C: PRNU + DCT MLP (noise residual + quantization forensics)
  D: Temporal Consistency CNN (frame-to-frame variation)
  E: FDI MLP (facial dynamics biomechanics)

Training additions:
  - Compression augmentation (random JPEG re-compression)
  - Multi-task auxiliary loss (predict manipulation type)

Usage:
  python main_v3.py          — full dataset
  python main_v3.py --pilot  — pilot test
"""

import os, sys, json, argparse
from pathlib import Path
import torch, h5py
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

DATA_ROOT_V2 = os.environ.get(
    "DATA_ROOT_V2",
    "D:/DO NOT TOUCH/deepfake_microexpression/data/FF++C23_preprocessed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot',      action='store_true')
    p.add_argument('--epochs',     type=int, default=60)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--patience',   type=int, default=20)
    args = p.parse_args()

    cache_dir  = os.environ.get("CACHE_DIR",  "cache")
    model_dir  = os.environ.get("MODEL_DIR",  "checkpoints")
    mode       = "pilot" if args.pilot else "full"
    cache_path = Path(cache_dir) / f"features_v3_{mode}_seed42.h5"
    run_dir    = Path(model_dir) / f"run_v3_{mode}"

    if not cache_path.exists():
        print(f"ERROR: v3 cache not found: {cache_path}")
        flag = '--pilot' if args.pilot else ''
        print(f"Run:  python scripts/precompute_features_v3.py {flag}")
        sys.exit(1)

    with h5py.File(str(cache_path), 'r') as f:
        prnu_dim = int(f['train']['prnu'].shape[1])  # 30
        dct_dim  = int(f['train']['dct'].shape[1])   # 48
        fdi_dim  = int(f['train']['fdi'].shape[1])   # 42

    prnu_dct_dim = prnu_dim + dct_dim  # 78

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'raw_model':            'efficientnet_b3',
        'spec_model':           'efficientnet_b0',
        'prnu_dct_dim':         prnu_dct_dim,
        'fdi_dim':              fdi_dim,
        'fusion_dim':           512,
        'fusion_dropout':       0.3,
        'efficientnet_dropout': 0.4,
        'n_manip_types':        7,
        'batch_size':           args.batch_size,
        'epochs':               args.epochs,
        'lr_backbone':          3e-5,
        'lr_heads':             3e-4,
        'weight_decay':         1e-4,
        'label_smoothing':      0.1,
        'grad_clip':            1.0,
        'mixed_precision':      True,
        'early_stop_patience':  args.patience,
    }

    print(f"\n{'='*60}")
    print(f"  Deepfake Detector v3 — Novel Five-Branch Architecture")
    print(f"{'='*60}")
    print(f"  Mode        : {mode}")
    print(f"  Device      : {device}")
    print(f"  Cache       : {cache_path}")
    print(f"  Run dir     : {run_dir}")
    print(f"  Branch C    : PRNU({prnu_dim}) + DCT({dct_dim}) = {prnu_dct_dim}-dim")
    print(f"  Branch E    : FDI {fdi_dim}-dim")
    print(f"  Augment     : JPEG compression + color jitter + random erasing")
    print(f"  Multi-task  : BCE + 0.3 × ManipType CrossEntropy")

    from models.detector_v3 import build_detector_v3
    model  = build_detector_v3(config).to(device)
    counts = model.count_parameters()
    print(f"\n  Parameters:")
    for k,v in counts.items():
        print(f"    {k:<25}: {v/1e6:.2f}M")

    from data.dataset_v3 import build_v3_dataloaders
    tr_loader, va_loader, te_loader = build_v3_dataloaders(
        str(cache_path), DATA_ROOT_V2,
        batch_size=args.batch_size, num_workers=2)

    model.freeze_raw_backbone()
    print(f"\n  B3 backbone frozen (unfreezes at epoch {5})")

    from train.trainer_v3 import TrainerV3
    trainer = TrainerV3(
        model=model, config=config,
        train_loader=tr_loader, val_loader=va_loader,
        test_loader=te_loader,
        model_dir=str(run_dir), device=device)

    trainer.train()
    m = trainer.evaluate_test_set()
    json.dump(m, open(run_dir/'test_results_v3.json','w'), indent=2)
    print(f"\n  TensorBoard: tensorboard --logdir {run_dir/'logs'}\n")


if __name__ == "__main__":
    main()