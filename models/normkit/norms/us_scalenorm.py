import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import numbers

__all__ = ['ScaleNorm']


class ScaleNorm(nn.Module):
    """ScaleNorm"""
    def __init__(self, num_features, eps=1e-5):
        super(ScaleNorm, self).__init__()
        self.scale = nn.Parameter(torch.tensor(num_features ** 0.5)*torch.ones(num_features))
        self.eps = eps
        self.num_features = num_features

    def forward(self, x, pad_mask=None, is_encoder=False):

        norm = self.scale.view(1,1,self.num_features) / torch.norm(x, dim=-1, keepdim=True).clamp(min=self.eps)
        return x * norm
