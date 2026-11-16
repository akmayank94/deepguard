"""
main.py — hybrid three-branch deepfake detector

Three inputs per frame:
  1. raw_rgb     — original face crop (ImageNet-normed) → raw EfficientNet
  2. diff_image  — inter-channel difference map         → spectral EfficientNet
  3. handcrafted — energy+DCCM feature vector           → MLP branch

No cache recompute. Uses existing features_full_seed42.h5 cache.
Raw frames loaded live from DATA_ROOT during training.
"""
import os, sys, json, argparse
from pathlib import Path
import torch, h5py
from dotenv import load_dotenv
load_dotenv(override=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pilot',      action='store_true')
    p.add_argument('--eval_only',  action='store_true')
    p.add_argument('--checkpoint', type=str, default=None)
    p.add_argument('--epochs',     type=int, default=60)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--patience',   type=int, default=15)
    args = p.parse_args()

    data_root = os.environ.get("DATA_ROOT", "")
    cache_dir = os.environ.get("CACHE_DIR", str(Path(__file__).parent/"cache"))
    model_dir = os.environ.get("MODEL_DIR", str(Path(__file__).parent/"checkpoints"))

    if not data_root:
        print("ERROR: DATA_ROOT not set in .env"); sys.exit(1)

    mode       = "pilot" if args.pilot else "full"
    cache_path = Path(cache_dir) / f"features_{mode}_seed42.h5"
    run_dir    = Path(model_dir) / f"run_{mode}_hybrid"

    if not cache_path.exists():
        print(f"ERROR: Cache not found: {cache_path}"); sys.exit(1)

    with h5py.File(str(cache_path), 'r') as f:
        mlp_dim = int(f['train']['handcrafted'].shape[1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = {
        'mlp_input_dim':           mlp_dim,
        'mlp_hidden_dims':         [512, 256],
        'mlp_dropouts':            [0.6, 0.5],
        'fusion_dim':              512,
        'fusion_dropout':          0.3,
        'efficientnet_dropout':    0.5,
        'efficientnet_pretrained': True,
        'batch_size':              args.batch_size,
        'epochs':                  args.epochs,
        'lr_backbone':             5e-5,
        'lr_heads':                5e-4,
        'weight_decay':            1e-4,
        'label_smoothing':         0.1,
        'grad_clip':               1.0,
        'mixed_precision':         True,
        'early_stop_patience':     args.patience,
    }

    print(f"\n{'='*55}")
    print(f"  Hybrid Deepfake Detector — 3 branches")
    print(f"{'='*55}")
    print(f"  Mode       : {mode}")
    print(f"  Device     : {device}")
    print(f"  Data root  : {data_root}")
    print(f"  Cache      : {cache_path}")
    print(f"  Run dir    : {run_dir}")

    from models.detector import build_detector
    model = build_detector(config).to(device)
    c = model.count_parameters()
    print(f"  Params     : {c['total']/1e6:.2f}M total")

    from data.dataset_hybrid import build_hybrid_dataloaders
    tr_loader, va_loader, te_loader = build_hybrid_dataloaders(
        str(cache_path), data_root,
        batch_size=args.batch_size, num_workers=2)

    if args.eval_only:
        ckpt_path = args.checkpoint or str(run_dir/'checkpoint_best.pt')
        ck = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(ck['model'])
        from train.trainer import Trainer
        t = Trainer(model, config, tr_loader, va_loader, te_loader,
                    str(run_dir), device)
        t.evaluate_test_set()
        return

    model.freeze_raw_backbone()
    print(f"  Raw backbone frozen (unfreezes at epoch 5)")

    from train.trainer import Trainer
    trainer = Trainer(model, config, tr_loader, va_loader, te_loader,
                      str(run_dir), device)
    trainer.train()
    m = trainer.evaluate_test_set()
    json.dump(m, open(run_dir/'test_results.json','w'), indent=2)
    print(f"\n  TensorBoard: tensorboard --logdir {run_dir/'logs'}\n")


if __name__ == "__main__":
    main()