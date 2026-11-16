"""models/efficientnet_branch.py — final"""
import torch.nn as nn
import timm


class EfficientNetBranch(nn.Module):
    def __init__(self, pretrained=True, dropout=0.3, freeze_bn=False):
        super().__init__()
        self.backbone = timm.create_model(
            'efficientnet_b0', pretrained=pretrained,
            num_classes=0, global_pool='avg')
        self.dropout   = nn.Dropout(p=dropout)
        self.output_dim = 1280

    def forward(self, x):
        return self.dropout(self.backbone(x))

    def get_parameter_groups(self, lr_backbone, lr_head):
        return [{'params': self.backbone.parameters(), 'lr': lr_backbone},
                {'params': self.dropout.parameters(),  'lr': lr_backbone}]