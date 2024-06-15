import torch.nn as nn
import torch


def model_device(m: nn.Module):
    return next(iter(m.parameters())).device


def model_numel(m: nn.Module, requires_grad=False):
    if requires_grad:
        return sum(p.numel() for p in m.parameters() if p.requires_grad)
    else:
        return sum(p.numel() for p in m.parameters())
