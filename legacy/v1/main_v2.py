"""
main_v2.py — Five-branch detector with EfficientNet-B3

Usage:
  python main_v2.py          — full dataset
  python main_v2.py --pilot  — pilot
"""
import os, sys, json, argparse
from pathlib import Path
import torch, h5py
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')

NEW_DATA_ROOT = os.environ.get(
    "DATA_ROOT_V2",
    "D:/DO NOT TOUCH/deepfake_microexpression/data/FF++C23_preprocessed")


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot',      action='store_true')
    p.add_argument('--eval_only',  action='store_true')
    p.add_argument('--epochs',     type=int, default=60)
    p.add_argument('--batch_size', type=int, default=12)
    p.add_argument('--patience',   type=int, default=20)
    args = p.parse_args()

    cache_dir  = os.environ.get("CACHE_DIR",  "cache")
    model_dir  = os.environ.get("MODEL_DIR",  "checkpoints")
    mode       = "pilot" if args.pilot else "full"
    cache_path = Path(cache_dir) / f"features_v2_{mode}_seed42.h5"
    run_dir    = Path(model_dir) / f"run_v2_{mode}_b3"

    if not cache_path.exists():
        print(f"ERROR: Cache not found: {cache_path}")
        sys.exit(1)

    with h5py.File(str(cache_path), 'r') as f:
        mlp_dim  = int(f['train']['handcrafted'].shape[1])
        prnu_dim = int(f['train']['prnu'].shape[1])
        rppg_dim = int(f['train']['rppg'].shape[1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'raw_model':            'efficientnet_b3',
        'spec_model':           'efficientnet_b0',
        'mlp_input_dim':        mlp_dim,
        'prnu_dim':             prnu_dim,
        'rppg_dim':             rppg_dim,
        'mlp_hidden_dims':      [512, 256],
        'mlp_dropouts':         [0.5, 0.4],
        'fusion_dim':           512,
        'fusion_dropout':       0.3,
        'efficientnet_dropout': 0.4,
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
    print(f"  Five-Branch Hybrid v2 — EfficientNet-B3 raw branch")
    print(f"{'='*60}")
    print(f"  Mode      : {mode}")
    print(f"  Device    : {device}")
    print(f"  Cache     : {cache_path}")
    print(f"  Run dir   : {run_dir}")
    print(f"  Batch sz  : {args.batch_size}")

    from models.detector_v2 import build_detector_v2
    model  = build_detector_v2(config).to(device)
    counts = model.count_parameters()
    print(f"\n  Parameters:")
    for k,v in counts.items():
        print(f"    {k:<24}: {v/1e6:.2f}M")

    from data.dataset_v2 import build_v2_dataloaders
    tr_loader, va_loader, te_loader = build_v2_dataloaders(
        str(cache_path), NEW_DATA_ROOT,
        batch_size=args.batch_size, num_workers=2)

    model.freeze_raw_backbone()
    print(f"\n  B3 backbone frozen (unfreezes at epoch 5)")

    from train.trainer_v2 import TrainerV2
    trainer = TrainerV2(
        model=model, config=config,
        train_loader=tr_loader, val_loader=va_loader,
        test_loader=te_loader,
        model_dir=str(run_dir), device=device)

    trainer.train()
    m = trainer.evaluate_test_set()
    json.dump(m, open(run_dir/'test_results.json','w'), indent=2)
    print(f"\n  TensorBoard: tensorboard --logdir {run_dir/'logs'}\n")


if __name__ == "__main__":
    main()