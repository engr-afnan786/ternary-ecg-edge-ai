"""Layerwise TWN-style forward quantization and a bounded identity STE."""
import torch
from torch import nn


@torch.no_grad()
def components(weight):
    """Return integer codes, scalar reconstruction scale, and scalar threshold."""
    magnitude = weight.abs()
    delta = 0.7 * magnitude.mean()
    active = magnitude > delta
    codes = (weight.sign() * active).to(torch.int8)
    alpha = (magnitude * active).sum() / active.sum().clamp_min(1)
    return codes, alpha, delta


class _EffectiveWeight(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight):
        codes, alpha, _ = components(weight)
        ctx.save_for_backward(weight)
        return codes.to(weight.dtype) * alpha

    @staticmethod
    def backward(ctx, upstream):
        (weight,) = ctx.saved_tensors
        return upstream * (weight.abs() <= 1)


class TernaryWeight(nn.Module):
    def forward(self, weight):
        return _EffectiveWeight.apply(weight)
