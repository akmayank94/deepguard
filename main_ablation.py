"""
main_ablation.py — Ablation Study

Trains three reduced model variants to prove each branch contributes.
Each variant saves to its own folder — nothing overwrites your main model.

Run order:
    python main_ablation.py --variant raw_only
    python main_ablation.py --variant spectral_hc_only
    python main_ablation.py --variant full  (optional, uses existing checkpoint)

Saved to:
    checkpoints/ablation_raw_only/
    checkpoints/ablation_spectral_hc_only/
    checkpoints/ablation_full/   (same as run_full_hybrid for reference)

Results saved to:
    checkpoints/ablation_results.json  (all three variants side by side)
"""

import os, sys, json, argparse
from pathlib import Path
import torch
import h5py
from dotenv import load_dotenv
load_dotenv(override=True)
sys.path.insert(0, '.')


# ── ablation model variants ──────────────────────────────────────────────────

class RawOnlyDetector(torch.nn.Module):
    """Branch A only: raw RGB EfficientNet. No spectral, no handcrafted."""
    def __init__(self, dropout=0.5):
        super().__init__()
        import timm
        self.backbone   = timm.create_model('efficientnet_b0', pretrained=True,
                          num_classes=0, global_pool='avg')
        self.dropout    = torch.nn.Dropout(p=dropout)
        self.classifier = torch.nn.Sequential(
            torch.nn.Linear(1280, 256), torch.nn.GELU(),
            torch.nn.Dropout(0.3), torch.nn.Linear(256, 1))

    def forward(self, raw_rgb, diff_image, handcrafted, return_gates=False):
        feat   = self.dropout(self.backbone(raw_rgb))
        logits = self.classifier(feat).squeeze(1)
        if return_gates:
            # Return dummy gates for compatibility with trainer
            gates = torch.ones(raw_rgb.size(0), 3, device=raw_rgb.device) / 3
            return logits, gates
        return logits

    def get_optimizer_param_groups(self, lr_backbone=1e-4, lr_heads=1e-3):
        return [
            {'params': self.backbone.parameters(),   'lr': lr_backbone, 'name': 'backbone'},
            {'params': self.dropout.parameters(),    'lr': lr_backbone, 'name': 'dropout'},
            {'params': self.classifier.parameters(), 'lr': lr_heads,    'name': 'classifier'},
        ]

    def freeze_raw_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad_(False)

    def unfreeze_raw_backbone(self):
        for p in self.backbone.parameters(): p.requires_grad_(True)

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total}


class SpectralHCDetector(torch.nn.Module):
    """Branch B + C only: spectral EfficientNet + handcrafted MLP. No raw RGB."""
    def __init__(self, mlp_input_dim=9369, dropout=0.5):
        super().__init__()
        import timm
        import torch.nn as nn
        # Spectral branch
        self.spec_backbone = timm.create_model('efficientnet_b0', pretrained=False,
                             num_classes=0, global_pool='avg')
        self.spec_dropout  = nn.Dropout(p=dropout)
        # Handcrafted branch
        self.input_bn = nn.BatchNorm1d(mlp_input_dim)
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 512), nn.BatchNorm1d(512), nn.GELU(), nn.Dropout(0.6),
            nn.Linear(512, 256), nn.BatchNorm1d(256), nn.GELU(), nn.Dropout(0.5))
        # Fusion
        import torch.nn.functional as F
        self.proj_spec = nn.Sequential(nn.Linear(1280,512), nn.LayerNorm(512), nn.GELU())
        self.proj_hc   = nn.Sequential(nn.Linear(256,  512), nn.LayerNorm(512), nn.GELU())
        self.gate      = nn.Sequential(nn.Linear(1024,128), nn.ReLU(), nn.Linear(128,2))
        torch.nn.init.zeros_(self.gate[-1].weight)
        torch.nn.init.zeros_(self.gate[-1].bias)
        self.classifier = nn.Sequential(
            nn.Linear(512,256), nn.GELU(), nn.Dropout(0.3), nn.Linear(256,1))

    def forward(self, raw_rgb, diff_image, handcrafted, return_gates=False):
        import torch.nn.functional as F
        fb = self.spec_dropout(self.spec_backbone(diff_image))
        fc = self.mlp(self.input_bn(handcrafted))
        pb = self.proj_spec(fb)
        pc = self.proj_hc(fc)
        gates = F.softmax(self.gate(torch.cat([pb,pc],dim=1)), dim=1)
        fused  = gates[:,0:1]*pb + gates[:,1:2]*pc
        logits = self.classifier(fused).squeeze(1)
        if return_gates:
            # Pad to 3 gates for trainer compatibility
            pad = torch.zeros(gates.size(0),1,device=gates.device)
            return logits, torch.cat([pad, gates], dim=1)
        return logits

    def get_optimizer_param_groups(self, lr_backbone=1e-4, lr_heads=1e-3):
        return [
            {'params': self.spec_backbone.parameters(), 'lr': lr_heads,    'name': 'spec'},
            {'params': self.spec_dropout.parameters(),  'lr': lr_heads,    'name': 'spec_do'},
            {'params': self.input_bn.parameters(),      'lr': lr_heads,    'name': 'hc_bn'},
            {'params': self.mlp.parameters(),           'lr': lr_heads,    'name': 'hc_mlp'},
            {'params': self.proj_spec.parameters(),     'lr': lr_heads,    'name': 'proj_s'},
            {'params': self.proj_hc.parameters(),       'lr': lr_heads,    'name': 'proj_h'},
            {'params': self.gate.parameters(),          'lr': lr_heads,    'name': 'gate'},
            {'params': self.classifier.parameters(),    'lr': lr_heads,    'name': 'cls'},
        ]

    def freeze_raw_backbone(self): pass   # no raw backbone
    def unfreeze_raw_backbone(self): pass

    def count_parameters(self):
        total = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total}


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--variant', required=True,
                   choices=['raw_only', 'spectral_hc_only', 'full'],
                   help='Which ablation variant to train')
    p.add_argument('--epochs',     type=int, default=60)
    p.add_argument('--batch_size', type=int, default=24)
    p.add_argument('--patience',   type=int, default=15)
    args = p.parse_args()

    data_root = os.environ.get("DATA_ROOT", "")
    cache_dir = os.environ.get("CACHE_DIR", "cache")
    model_dir = os.environ.get("MODEL_DIR", "checkpoints")

    cache_path = Path(cache_dir) / "features_full_seed42.h5"
    run_dir    = Path(model_dir) / f"ablation_{args.variant}"

    if not cache_path.exists():
        print(f"ERROR: Cache not found: {cache_path}"); sys.exit(1)
    if not data_root:
        print("ERROR: DATA_ROOT not set in .env"); sys.exit(1)

    with h5py.File(str(cache_path), 'r') as f:
        mlp_dim = int(f['train']['handcrafted'].shape[1])

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"\n{'='*55}")
    print(f"  Ablation Study — variant: {args.variant}")
    print(f"{'='*55}")
    print(f"  Run dir : {run_dir}")
    print(f"  Device  : {device}")

    # Build model variant
    if args.variant == 'raw_only':
        model = RawOnlyDetector(dropout=0.5).to(device)
        print(f"  Model   : Raw RGB EfficientNet only")
    elif args.variant == 'spectral_hc_only':
        model = SpectralHCDetector(mlp_input_dim=mlp_dim, dropout=0.5).to(device)
        print(f"  Model   : Spectral + Handcrafted only (no raw RGB)")
    else:  # full
        from models.detector import build_detector
        config = {'mlp_input_dim': mlp_dim,
                  'mlp_hidden_dims': [512,256], 'mlp_dropouts': [0.6,0.5],
                  'fusion_dim': 512, 'fusion_dropout': 0.3,
                  'efficientnet_dropout': 0.5, 'efficientnet_pretrained': True}
        model = build_detector(config).to(device)
        print(f"  Model   : Full three-branch (same as main training)")

    counts = model.count_parameters()
    print(f"  Params  : {counts['total']/1e6:.2f}M")

    config_train = {
        'mlp_input_dim': mlp_dim,
        'batch_size':    args.batch_size,
        'epochs':        args.epochs,
        'lr_backbone':   5e-5,
        'lr_heads':      5e-4,
        'weight_decay':  1e-4,
        'label_smoothing': 0.1,
        'grad_clip':     1.0,
        'mixed_precision': True,
        'early_stop_patience': args.patience,
        'variant': args.variant,
    }

    from data.dataset_hybrid import build_hybrid_dataloaders
    tr_loader, va_loader, te_loader = build_hybrid_dataloaders(
        str(cache_path), data_root,
        batch_size=args.batch_size, num_workers=2)

    model.freeze_raw_backbone()

    from train.trainer import Trainer
    trainer = Trainer(model, config_train, tr_loader, va_loader,
                      te_loader, str(run_dir), device)
    trainer.train()
    m = trainer.evaluate_test_set()

    # Save to ablation results file (append, do not overwrite other variants)
    ablation_path = Path(model_dir) / "ablation_results.json"
    ablation_data = json.load(open(ablation_path)) if ablation_path.exists() else {}
    ablation_data[args.variant] = {
        'test_auc':  m['auc'],
        'test_f1':   m['f1'],
        'test_acc':  m['acc'],
        'val_auc':   trainer.best_val_auc,
        'best_epoch': trainer.best_epoch,
        'params':    counts['total'],
        'per_type':  m.get('manip_aucs', {}),
    }
    json.dump(ablation_data, open(ablation_path,'w'), indent=2)
    print(f"\n  Ablation results saved to: {ablation_path}")
    print(f"  Run dir: {run_dir}")
    print(f"\n  Summary — variant={args.variant}:")
    print(f"    Val  AUC: {trainer.best_val_auc:.4f}")
    print(f"    Test AUC: {m['auc']:.4f}  F1: {m['f1']:.4f}  Acc: {m['acc']:.4f}")


if __name__ == "__main__":
    main()