"""
models/handcrafted_branch.py — final

BatchNorm1d on input handles dataset-level normalization during training.
This replaces per-image standardization in feature extraction.
"""
import torch.nn as nn


class HandcraftedBranch(nn.Module):
    def __init__(self, input_dim=9261, hidden_dims=None, dropouts=None):
        super().__init__()
        if hidden_dims is None: hidden_dims = [512, 256]
        if dropouts    is None: dropouts    = [0.5, 0.4]
        self.output_dim = hidden_dims[-1]
        self.input_bn   = nn.BatchNorm1d(input_dim)
        layers, in_dim = [], input_dim
        for h, d in zip(hidden_dims, dropouts):
            layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h),
                       nn.GELU(), nn.Dropout(p=d)]
            in_dim = h
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return self.mlp(self.input_bn(x))

    def get_parameter_groups(self, lr):
        return [{'params': self.parameters(), 'lr': lr}]