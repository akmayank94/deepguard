"""
models/detector.py — three-branch with high dropout to prevent overfitting
"""
import torch, torch.nn as nn, torch.nn.functional as F
import timm


class EfficientNetBranch(nn.Module):
    def __init__(self, pretrained=True, dropout=0.5):
        super().__init__()
        self.backbone   = timm.create_model('efficientnet_b0',
                          pretrained=pretrained, num_classes=0, global_pool='avg')
        self.dropout    = nn.Dropout(p=dropout)
        self.output_dim = 1280
    def forward(self, x):
        return self.dropout(self.backbone(x))


class HandcraftedBranch(nn.Module):
    def __init__(self, input_dim=9369, hidden_dims=None, dropouts=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [512, 256]
        if dropouts    is None: dropouts    = [0.6, 0.5]
        self.output_dim = hidden_dims[-1]
        self.input_bn   = nn.BatchNorm1d(input_dim)
        layers, d = [], input_dim
        for h, dr in zip(hidden_dims, dropouts):
            layers += [nn.Linear(d,h), nn.BatchNorm1d(h), nn.GELU(), nn.Dropout(dr)]
            d = h
        self.mlp = nn.Sequential(*layers)
    def forward(self, x):
        return self.mlp(self.input_bn(x))


class ThreeBranchFusion(nn.Module):
    def __init__(self, dim_a=1280, dim_b=1280, dim_c=256,
                 fusion_dim=512, dropout=0.3):
        super().__init__()
        self.proj_a = nn.Sequential(nn.Linear(dim_a, fusion_dim),
                                    nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_b = nn.Sequential(nn.Linear(dim_b, fusion_dim),
                                    nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_c = nn.Sequential(nn.Linear(dim_c, fusion_dim),
                                    nn.LayerNorm(fusion_dim), nn.GELU())
        self.gate   = nn.Sequential(nn.Linear(fusion_dim*3, 256), nn.ReLU(),
                                    nn.Linear(256, 3))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim//2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(fusion_dim//2, 1))

    def forward(self, fa, fb, fc, return_gates=False):
        pa, pb, pc = self.proj_a(fa), self.proj_b(fb), self.proj_c(fc)
        gates  = F.softmax(self.gate(torch.cat([pa,pb,pc],1)), dim=1)
        fused  = gates[:,0:1]*pa + gates[:,1:2]*pb + gates[:,2:3]*pc
        logits = self.classifier(fused).squeeze(1)
        return (logits, gates) if return_gates else logits


class DeepfakeDetector(nn.Module):
    def __init__(self, efficientnet_pretrained=True, efficientnet_dropout=0.5,
                 mlp_input_dim=9369, mlp_hidden_dims=None, mlp_dropouts=None,
                 fusion_dim=512, fusion_dropout=0.3):
        super().__init__()
        if mlp_hidden_dims is None: mlp_hidden_dims = [512, 256]
        if mlp_dropouts    is None: mlp_dropouts    = [0.6, 0.5]
        self.branch_raw  = EfficientNetBranch(pretrained=efficientnet_pretrained,
                                              dropout=efficientnet_dropout)
        self.branch_spec = EfficientNetBranch(pretrained=False,
                                              dropout=efficientnet_dropout)
        self.branch_hc   = HandcraftedBranch(mlp_input_dim, mlp_hidden_dims, mlp_dropouts)
        self.fusion = ThreeBranchFusion(
            self.branch_raw.output_dim, self.branch_spec.output_dim,
            self.branch_hc.output_dim, fusion_dim, fusion_dropout)

    def forward(self, raw_rgb, diff_image, handcrafted, return_gates=False):
        return self.fusion(self.branch_raw(raw_rgb),
                           self.branch_spec(diff_image),
                           self.branch_hc(handcrafted),
                           return_gates=return_gates)

    def get_optimizer_param_groups(self, lr_backbone=1e-4, lr_heads=1e-3):
        return [
            {'params': self.branch_raw.parameters(),  'lr': lr_backbone, 'name': 'raw_effnet'},
            {'params': self.branch_spec.parameters(), 'lr': lr_heads,    'name': 'spec_effnet'},
            {'params': self.branch_hc.parameters(),   'lr': lr_heads,    'name': 'hc_mlp'},
            {'params': self.fusion.parameters(),      'lr': lr_heads,    'name': 'fusion'},
        ]

    def freeze_raw_backbone(self):
        for p in self.branch_raw.backbone.parameters(): p.requires_grad_(False)

    def unfreeze_raw_backbone(self):
        for p in self.branch_raw.backbone.parameters(): p.requires_grad_(True)

    def count_parameters(self):
        def _c(m): return sum(p.numel() for p in m.parameters() if p.requires_grad)
        return {'branch_raw': _c(self.branch_raw), 'branch_spec': _c(self.branch_spec),
                'branch_hc': _c(self.branch_hc), 'fusion': _c(self.fusion), 'total': _c(self)}


def build_detector(config=None):
    if config is None: config = {}
    return DeepfakeDetector(
        efficientnet_pretrained=config.get('efficientnet_pretrained', True),
        efficientnet_dropout   =config.get('efficientnet_dropout',    0.5),
        mlp_input_dim          =config.get('mlp_input_dim',           9369),
        mlp_hidden_dims        =config.get('mlp_hidden_dims',         [512, 256]),
        mlp_dropouts           =config.get('mlp_dropouts',            [0.6, 0.5]),
        fusion_dim             =config.get('fusion_dim',              512),
        fusion_dropout         =config.get('fusion_dropout',          0.3),
    )