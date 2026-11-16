"""
train/trainer_v3.py — v3 trainer with multi-task auxiliary loss

Multi-task loss = BCE(real/fake) + 0.3 × CE(manipulation type)
Forces the model to learn type-specific features, improving generalization.
"""

import sys, time, json
from pathlib import Path
from collections import defaultdict

import torch, torch.nn as nn
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))
from models.losses import build_loss

UNFREEZE_EPOCH    = 5
AUX_LOSS_WEIGHT   = 0.3   # weight of manipulation-type classification loss


class TrainerV3:
    def __init__(self, model, config, train_loader, val_loader,
                 test_loader, model_dir, device):
        self.model        = model
        self.config       = config
        self.train_loader = train_loader
        self.val_loader   = val_loader
        self.test_loader  = test_loader
        self.model_dir    = Path(model_dir)
        self.device       = device
        self.model_dir.mkdir(parents=True, exist_ok=True)

        self.criterion     = build_loss(
            config.get('label_smoothing', 0.1)).to(device)
        self.type_criterion = nn.CrossEntropyLoss().to(device)

        groups = model.get_optimizer_param_groups(
            lr_backbone=config.get('lr_backbone', 3e-5),
            lr_heads   =config.get('lr_heads',    3e-4))
        self.optimizer = torch.optim.AdamW(
            groups, weight_decay=config.get('weight_decay', 1e-4))
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=1e-6)
        self.scaler    = GradScaler(enabled=config.get('mixed_precision', True))
        self.writer    = SummaryWriter(str(self.model_dir/'logs'))

        self.best_val_auc = 0.0
        self.best_epoch   = 0
        self.no_improve   = 0
        self.history      = []
        self.max_epochs   = config.get('epochs', 60)
        self.grad_clip    = config.get('grad_clip', 1.0)
        self.patience     = config.get('early_stop_patience', 20)

        print(f"  TrainerV3 ready | train={len(train_loader)} "
              f"val={len(val_loader)} batches")
        print(f"  Multi-task loss weight: {AUX_LOSS_WEIGHT}")
        print(f"  Backbone unfreezes at epoch {UNFREEZE_EPOCH}")

    def _run(self, loader, train=True, desc=""):
        self.model.train() if train else self.model.eval()
        total_loss = 0.0
        all_logits, all_labels, all_types = [], [], []
        gate_accum = []

        ctx = torch.enable_grad() if train else torch.no_grad()
        with ctx:
            for batch in tqdm(loader, desc=f"  {desc}",
                              ncols=72, leave=False, unit="b"):
                rr   = batch['raw_rgb'].to(self.device,      non_blocking=True)
                di   = batch['diff_image'].to(self.device,    non_blocking=True)
                pd   = batch['prnu_dct'].to(self.device,      non_blocking=True)
                ts   = batch['temporal_std'].to(self.device,  non_blocking=True)
                fdi  = batch['fdi'].to(self.device,           non_blocking=True)
                lbl  = batch['label'].float().to(self.device, non_blocking=True)
                tidx = batch['type_idx'].to(self.device,      non_blocking=True)

                if train: self.optimizer.zero_grad(set_to_none=True)

                with autocast(enabled=self.scaler.is_enabled()):
                    if train:
                        logits, gates, type_logits = self.model(
                            rr, di, pd, ts, fdi,
                            return_gates=True, return_type=True)
                        gate_accum.append(gates.detach().cpu().numpy())

                        # Multi-task loss
                        bce_loss  = self.criterion(logits, lbl)
                        type_loss = self.type_criterion(type_logits, tidx)
                        loss      = bce_loss + AUX_LOSS_WEIGHT * type_loss
                    else:
                        logits = self.model(
                            rr, di, pd, ts, fdi,
                            return_gates=False, return_type=False)
                        loss = self.criterion(logits, lbl)

                if train:
                    self.scaler.scale(loss).backward()
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(),
                                             self.grad_clip)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()

                total_loss += loss.item()
                all_logits.append(logits.detach().cpu())
                all_labels.append(lbl.detach().cpu())
                all_types.extend(batch['manip_type'])

        logits_np = torch.cat(all_logits).numpy()
        labels_np = torch.cat(all_labels).numpy()
        probs_np  = 1/(1+np.exp(-logits_np))
        preds_np  = (probs_np >= 0.5).astype(int)

        m = {
            'loss': total_loss/len(loader),
            'acc':  float(accuracy_score(labels_np, preds_np)),
            'f1':   float(f1_score(labels_np, preds_np, zero_division=0)),
            'auc':  float(roc_auc_score(labels_np, probs_np)),
            'manip_aucs': {},
        }
        for mt in set(all_types):
            if mt == 'real': continue
            idx = [i for i,t in enumerate(all_types) if t in (mt,'real')]
            if len(idx) < 10: continue
            ml, mp = labels_np[idx], probs_np[idx]
            if len(np.unique(ml)) < 2: continue
            try: m['manip_aucs'][mt] = float(roc_auc_score(ml, mp))
            except: pass
        if gate_accum:
            g = np.mean(np.vstack(gate_accum), axis=0)
            m['gates'] = g.tolist()
        return m

    def _save(self, epoch, val_auc, tag='best'):
        torch.save({'epoch':epoch, 'val_auc':val_auc,
                    'model':self.model.state_dict(),
                    'config':self.config},
                   self.model_dir/f'checkpoint_{tag}.pt')

    def _load_best(self):
        p = self.model_dir/'checkpoint_best.pt'
        if p.exists():
            ck = torch.load(p, map_location=self.device)
            self.model.load_state_dict(ck['model'])
            return ck['val_auc']
        return 0.0

    def train(self):
        print(f"\n{'='*60}\n  Training v3\n{'='*60}")
        t0 = time.time()
        for epoch in range(1, self.max_epochs+1):
            if epoch == UNFREEZE_EPOCH:
                self.model.unfreeze_raw_backbone()
                print(f"  B3 backbone unfrozen at epoch {epoch}")

            t_ep = time.time()
            tr = self._run(self.train_loader, True,  f"Ep{epoch:03d}[tr]")
            va = self._run(self.val_loader,   False, f"Ep{epoch:03d}[va]")
            self.scheduler.step()

            g    = tr.get('gates', [0.2]*5)
            gstr = ','.join(f"{x:.2f}" for x in g)
            print(f"  Ep {epoch:03d}  tr={tr['auc']:.3f}  val={va['auc']:.3f}  "
                  f"f1={va['f1']:.3f}  acc={va['acc']:.3f}  "
                  f"g=[{gstr}]  [{time.time()-t_ep:.0f}s]")
            if va['manip_aucs']:
                s = ' '.join(f"{k[:4]}={v:.3f}"
                             for k,v in sorted(va['manip_aucs'].items()))
                print(f"         {s}")

            self.writer.add_scalars('AUC',
                {'train':tr['auc'],'val':va['auc']}, epoch)
            labels = ['raw_B3','spec_B0','prnu_dct','temporal','fdi']
            for i, gv in enumerate(g):
                self.writer.add_scalar(f'Gate/{labels[i]}', gv, epoch)

            if va['auc'] > self.best_val_auc + 1e-4:
                self.best_val_auc = va['auc']
                self.best_epoch   = epoch
                self.no_improve   = 0
                self._save(epoch, va['auc'], 'best')
                print(f"  ✓ best={va['auc']:.4f}")
            else:
                self.no_improve += 1
                if self.no_improve >= self.patience:
                    print(f"\n  Early stop at epoch {epoch}")
                    break

            if epoch % 5 == 0: self._save(epoch, va['auc'], 'latest')
            self.history.append({'ep':epoch,'tr':tr['auc'],
                                 'va':va['auc'],'gates':g})

        print(f"\n  Best val AUC: {self.best_val_auc:.4f} "
              f"at epoch {self.best_epoch}")
        print(f"  Time: {(time.time()-t0)/60:.1f} min")
        self.writer.close()
        json.dump(self.history,
                  open(self.model_dir/'history.json','w'), indent=2)
        return self.history

    def evaluate_test_set(self):
        best = self._load_best()
        print(f"\n  Test set (val_auc={best:.4f})")
        m = self._run(self.test_loader, False, "test")
        print(f"  acc={m['acc']:.4f}  f1={m['f1']:.4f}  auc={m['auc']:.4f}")
        for mt, auc in sorted(m['manip_aucs'].items()):
            print(f"    {mt:<22}: {auc:.4f}")
        return m