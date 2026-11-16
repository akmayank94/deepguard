import torch.nn.functional as F, torch.nn as nn
class LabelSmoothingBCE(nn.Module):
    def __init__(self, smoothing=0.1):
        super().__init__(); self.s = smoothing
    def forward(self, logits, labels):
        labels = labels.float()
        soft = labels*(1-self.s) + (1-labels)*self.s
        return F.binary_cross_entropy_with_logits(logits, soft)
def build_loss(smoothing=0.1): return LabelSmoothingBCE(smoothing)