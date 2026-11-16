"""
models/detector_v3.py — Complete five-branch detector

Branch A: EfficientNet-B3 on raw RGB (ImageNet pretrained)
Branch B: EfficientNet-B0 on spectral diff_image (from scratch)
Branch C: Combined PRNU + DCT MLP (camera noise + quantization forensics)
Branch D: Small CNN on temporal std map (frame-to-frame consistency)
Branch E: FDI MLP (facial dynamics biomechanics)

Plus: auxiliary manipulation-type classifier for multi-task learning.
      shared fusion features → predict 7 classes (real + 6 fake types)
      loss = bce_loss + 0.3 * cross_entropy_type_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm


class EfficientNetBranch(nn.Module):
    def __init__(self, model_name='efficientnet_b0',
                 pretrained=True, dropout=0.4):
        super().__init__()
        self.backbone = timm.create_model(
            model_name, pretrained=pretrained,
            num_classes=0, global_pool='avg')
        self.dropout = nn.Dropout(p=dropout)
        with torch.no_grad():
            self.output_dim = self.backbone(
                torch.zeros(1,3,224,224)).shape[1]

    def forward(self, x):
        return self.dropout(self.backbone(x))


class SmallCNN(nn.Module):
    """Lightweight CNN for 224x224x3 temporal std map."""
    def __init__(self, dropout=0.3):
        super().__init__()
        self.output_dim = 128
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32),
            nn.GELU(), nn.MaxPool2d(2),               # →112
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64),
            nn.GELU(), nn.MaxPool2d(2),               # →56
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.GELU(), nn.MaxPool2d(2),               # →28
            nn.Conv2d(128, 128, 3, padding=1), nn.BatchNorm2d(128),
            nn.GELU(), nn.AdaptiveAvgPool2d(1),
            nn.Flatten(), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class HandcraftedMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropouts, output_dim=128):
        super().__init__()
        self.output_dim = output_dim
        self.input_bn   = nn.BatchNorm1d(input_dim)
        layers, d = [], input_dim
        for h, dr in zip(hidden_dims, dropouts):
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(dr)]
            d = h
        layers += [nn.Linear(d, output_dim),
                   nn.BatchNorm1d(output_dim), nn.GELU()]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.input_bn(x))


class FiveBranchFusion(nn.Module):
    def __init__(self, dims, fusion_dim=512, dropout=0.3, n_types=7):
        super().__init__()
        self.projs = nn.ModuleList([
            nn.Sequential(nn.Linear(d, fusion_dim),
                          nn.LayerNorm(fusion_dim), nn.GELU())
            for d in dims])
        n = len(dims)
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim*n, 256), nn.ReLU(),
            nn.Linear(256, n))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim//2, 1))

        # Auxiliary: predict manipulation type (multi-task)
        self.type_classifier = nn.Sequential(
            nn.Linear(fusion_dim, 128), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(128, n_types))

    def forward(self, features, return_gates=False, return_type=False):
        projected = [proj(f) for proj,f in zip(self.projs, features)]
        gates  = F.softmax(self.gate(torch.cat(projected, dim=1)), dim=1)
        fused  = sum(gates[:,i:i+1]*p for i,p in enumerate(projected))
        logits = self.classifier(fused).squeeze(1)

        out = (logits, gates) if return_gates else logits

        if return_type:
            type_logits = self.type_classifier(fused)
            if return_gates:
                return logits, gates, type_logits
            return logits, type_logits

        return out


class DeepfakeDetectorV3(nn.Module):
    def __init__(self,
                 raw_model='efficientnet_b3',
                 spec_model='efficientnet_b0',
                 prnu_dct_dim=78,   # 30 prnu + 48 dct
                 fdi_dim=42,
                 fusion_dim=512,
                 fusion_dropout=0.3,
                 efficientnet_dropout=0.4,
                 n_manip_types=7):  # real + 6 fake types
        super().__init__()

        self.branch_raw  = EfficientNetBranch(raw_model,  True,
                                              efficientnet_dropout)
        self.branch_spec = EfficientNetBranch(spec_model, False,
                                              efficientnet_dropout)
        self.branch_c    = HandcraftedMLP(prnu_dct_dim,
                                          [128, 64], [0.4, 0.3],
                                          output_dim=64)
        self.branch_temp = SmallCNN(dropout=0.3)
        self.branch_fdi  = HandcraftedMLP(fdi_dim,
                                          [64, 64], [0.3, 0.2],
                                          output_dim=64)

        self.fusion = FiveBranchFusion(
            dims=[self.branch_raw.output_dim,
                  self.branch_spec.output_dim,
                  self.branch_c.output_dim,
                  self.branch_temp.output_dim,
                  self.branch_fdi.output_dim],
            fusion_dim=fusion_dim,
            dropout=fusion_dropout,
            n_types=n_manip_types)

    def forward(self, raw_rgb, diff_image, prnu_dct,
                temporal_std, fdi,
                return_gates=False, return_type=False):
        fa = self.branch_raw(raw_rgb)
        fb = self.branch_spec(diff_image)
        fc = self.branch_c(prnu_dct)
        fd = self.branch_temp(temporal_std)
        fe = self.branch_fdi(fdi)
        return self.fusion([fa,fb,fc,fd,fe],
                           return_gates=return_gates,
                           return_type=return_type)

    def get_optimizer_param_groups(self, lr_backbone=3e-5, lr_heads=3e-4):
        return [
            {'params': self.branch_raw.parameters(),
             'lr': lr_backbone, 'name': 'raw_b3'},
            {'params': self.branch_spec.parameters(),
             'lr': lr_heads,    'name': 'spec_b0'},
            {'params': self.branch_c.parameters(),
             'lr': lr_heads,    'name': 'prnu_dct'},
            {'params': self.branch_temp.parameters(),
             'lr': lr_heads,    'name': 'temporal'},
            {'params': self.branch_fdi.parameters(),
             'lr': lr_heads,    'name': 'fdi'},
            {'params': self.fusion.parameters(),
             'lr': lr_heads,    'name': 'fusion'},
        ]

    def freeze_raw_backbone(self):
        for p in self.branch_raw.backbone.parameters():
            p.requires_grad_(False)

    def unfreeze_raw_backbone(self):
        for p in self.branch_raw.backbone.parameters():
            p.requires_grad_(True)

    def count_parameters(self):
        def _c(m): return sum(p.numel() for p in m.parameters()
                              if p.requires_grad)
        return {
            'branch_raw(B3)':   _c(self.branch_raw),
            'branch_spec(B0)':  _c(self.branch_spec),
            'branch_c(PRNU+DCT)': _c(self.branch_c),
            'branch_temporal':  _c(self.branch_temp),
            'branch_fdi':       _c(self.branch_fdi),
            'fusion':           _c(self.fusion),
            'total':            _c(self),
        }


# Manipulation type to index mapping
MANIP_TYPE_TO_IDX = {
    'real': 0,
    'Deepfakes': 1,
    'Face2Face': 2,
    'FaceShifter': 3,
    'FaceSwap': 4,
    'NeuralTextures': 5,
    'DeepFakeDetection': 6,
}


def build_detector_v3(config=None):
    if config is None: config = {}
    return DeepfakeDetectorV3(
        raw_model           =config.get('raw_model',            'efficientnet_b3'),
        spec_model          =config.get('spec_model',           'efficientnet_b0'),
        prnu_dct_dim        =config.get('prnu_dct_dim',         78),
        fdi_dim             =config.get('fdi_dim',              42),
        fusion_dim          =config.get('fusion_dim',           512),
        fusion_dropout      =config.get('fusion_dropout',       0.3),
        efficientnet_dropout=config.get('efficientnet_dropout', 0.4),
        n_manip_types       =config.get('n_manip_types',        7),
    )