"""CAM, Grad-CAM and Grad-CAM++ for convolutional networks (and any module producing (B, C, H, W) activations).

References
  CAM         Zhou et al., CVPR 2016
  Grad-CAM    Selvaraju et al., ICCV 2017
  Grad-CAM++  Chattopadhay et al., WACV 2018 (closed form under S = exp(y), as in the paper and pytorch-grad-cam)
All functions return a (H, W) float tensor at the target layer's resolution for a single image (batch of 1).
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Callable, Optional

import torch
import torch.nn as nn


@contextmanager
def capture(layer: nn.Module, reshape: Optional[Callable] = None):
    """Record the forward output of `layer` and its gradient. `reshape` maps tokens → (B, C, H, W) if needed."""
    store = {}

    def fwd(_m, _i, out):
        out = out[0] if isinstance(out, tuple) else out
        store["act"] = out
        if out.requires_grad:
            out.register_hook(lambda g: store.__setitem__("grad", g))

    h = layer.register_forward_hook(fwd)
    try:
        yield store
    finally:
        h.remove()
    if reshape is not None:
        if "act" in store:
            store["act"] = reshape(store["act"])
        if "grad" in store:
            store["grad"] = reshape(store["grad"])


def _score(model, x, target_class):
    logits = model(x)
    return logits, logits[:, target_class].sum()


def cam_densenet(model: nn.Module, x: torch.Tensor, target_class: int) -> torch.Tensor:
    """Exact CAM for DenseNet161Classifier: M = Σ_k w_k^c A_k with A = ReLU(features) and w = classifier.weight[c]."""
    with torch.no_grad():
        a = model.forward_features(x)  # (1, 2208, 7, 7)
        w = model.classifier.weight[target_class]  # (2208,)
        return torch.einsum("k,bkhw->bhw", w, a)[0]


def gradcam(model: nn.Module, x: torch.Tensor, target_class: int, target_layer: nn.Module, reshape: Optional[Callable] = None,
            relu: bool = True) -> torch.Tensor:
    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)  # gradients flow to activations even if weights are frozen
    with torch.enable_grad():
        with capture(target_layer, reshape) as s:
            _, score = _score(model, x, target_class)
            score.backward()
    a, g = s["act"].detach(), s["grad"].detach()
    alpha = g.mean(dim=(2, 3), keepdim=True)
    m = (alpha * a).sum(1)[0]
    return torch.relu(m) if relu else m


def gradcampp(model: nn.Module, x: torch.Tensor, target_class: int, target_layer: nn.Module, reshape: Optional[Callable] = None) -> torch.Tensor:
    if not x.requires_grad:
        x = x.detach().clone().requires_grad_(True)
    with torch.enable_grad():
        with capture(target_layer, reshape) as s:
            _, score = _score(model, x, target_class)
            score.backward()
    a, g = s["act"].detach(), s["grad"].detach()
    g2, g3 = g ** 2, g ** 3
    sum_a = a.sum(dim=(2, 3), keepdim=True)
    denom = 2 * g2 + sum_a * g3
    alpha = torch.where(g != 0, g2 / torch.where(denom != 0, denom, torch.ones_like(denom)), torch.zeros_like(g))
    w = (alpha * torch.relu(g)).sum(dim=(2, 3), keepdim=True)
    return torch.relu((w * a).sum(1))[0]


def upsample(m: torch.Tensor, size=(224, 224)) -> torch.Tensor:
    return torch.nn.functional.interpolate(m[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0]


def normalise(m: torch.Tensor) -> torch.Tensor:
    lo, hi = m.min(), m.max()
    return (m - lo) / (hi - lo) if hi > lo else torch.zeros_like(m)
