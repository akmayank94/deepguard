"""
models/attention_fusion.py — final

Gate initialized to 0.5/0.5. LayerNorm on both projections ensures
equal magnitude going into the gate regardless of branch output scale.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class AttentionFusion(nn.Module):
    def __init__(self, dim_a=1280, dim_b=256, fusion_dim=512,
                 gate_hidden=128, dropout=0.2):
        super().__init__()
        self.proj_a = nn.Sequential(
            nn.Linear(dim_a, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.proj_b = nn.Sequential(
            nn.Linear(dim_b, fusion_dim), nn.LayerNorm(fusion_dim), nn.GELU())
        self.gate = nn.Sequential(
            nn.Linear(fusion_dim * 2, gate_hidden), nn.ReLU(),
            nn.Linear(gate_hidden, 2))
        nn.init.zeros_(self.gate[-1].weight)
        nn.init.zeros_(self.gate[-1].bias)
        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2), nn.GELU(),
            nn.Dropout(p=dropout), nn.Linear(fusion_dim // 2, 1))

    def forward(self, feat_a, feat_b, return_gates=False):
        pa = self.proj_a(feat_a)
        pb = self.proj_b(feat_b)
        gates = F.softmax(self.gate(torch.cat([pa, pb], dim=1)), dim=1)
        fused = gates[:, 0:1] * pa + gates[:, 1:2] * pb
        logits = self.classifier(fused).squeeze(1)
        return (logits, gates) if return_gates else logits