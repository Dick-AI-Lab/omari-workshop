"""Attribution and attention tools for timm VisionTransformer models (ViT-B/16 and the USF-MAE encoder).

  attention_maps        raw softmax(QKᵀ/√d) per block and head (forward hooks; disables fused attention)
  attention_rollout     Abnar & Zuidema (2020): R = Π_ℓ ½(Ā_ℓ + I), row-normalised; [CLS] row → 14×14
  grad_rollout          gradient-weighted rollout: Ā_ℓ = E_h[ReLU(∂y/∂A ⊙ A)] (simplified Chefer et al. 2021, no LRP)
  token_gradcam(++)     Grad-CAM(++) on patch tokens at the input of blocks[b].norm1 (or output of a block),
                        reshaped 196 → 14×14 with [CLS] dropped. For the *last* block, gradients reach patch tokens
                        only through attention to [CLS], which is why the norm1 input of the final block is used.
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Dict, List, Optional

import torch
import torch.nn as nn

from .cam import gradcam, gradcampp


def tokens_to_map(t: torch.Tensor, prefix: int = 1) -> torch.Tensor:
    """(B, 1+N, D) → (B, D, √N, √N)."""
    t = t[:, prefix:, :]
    n = t.shape[1]
    g = int(round(n ** 0.5))
    return t.reshape(t.shape[0], g, g, t.shape[2]).permute(0, 3, 1, 2)


class _Norm1Input(nn.Module):
    """Wrapper exposing a block's norm1 *input* as a hookable module output."""


@contextmanager
def _norm1_input_hook(block: nn.Module):
    store = {}

    def pre(_m, inp):
        t = inp[0]
        store["act"] = t
        if t.requires_grad:
            t.register_hook(lambda g: store.__setitem__("grad", g))

    h = block.norm1.register_forward_pre_hook(pre)
    try:
        yield store
    finally:
        h.remove()


def _token_cam(model, x, target_class, block: int, plus: bool) -> torch.Tensor:
    blk = model.blocks[block]
    prefix = getattr(model, "num_prefix_tokens", 1)
    with torch.enable_grad():
        with _norm1_input_hook(blk) as s:
            xx = x if x.requires_grad else x.detach().clone().requires_grad_(True)  # ensure graph reaches tokens even if weights are frozen
            logits = model(xx)
            logits[:, target_class].sum().backward()
    a = tokens_to_map(s["act"].detach(), prefix)
    g = tokens_to_map(s["grad"].detach(), prefix)
    if not plus:
        alpha = g.mean(dim=(2, 3), keepdim=True)
        return torch.relu((alpha * a).sum(1))[0]
    g2, g3 = g ** 2, g ** 3
    sum_a = a.sum(dim=(2, 3), keepdim=True)
    denom = 2 * g2 + sum_a * g3
    alpha = torch.where(g != 0, g2 / torch.where(denom != 0, denom, torch.ones_like(denom)), torch.zeros_like(g))
    w = (alpha * torch.relu(g)).sum(dim=(2, 3), keepdim=True)
    return torch.relu((w * a).sum(1))[0]


def token_gradcam(model, x, target_class: int, block: int = -1) -> torch.Tensor:
    return _token_cam(model, x, target_class, block, plus=False)


def token_gradcampp(model, x, target_class: int, block: int = -1) -> torch.Tensor:
    return _token_cam(model, x, target_class, block, plus=True)


@contextmanager
def capture_attention(model, keep_grad: bool = False):
    """Materialise attention matrices. Yields dict block → tensor (B, heads, T, T) (and grads if keep_grad)."""
    attn: Dict[int, torch.Tensor] = {}
    grads: Dict[int, torch.Tensor] = {}
    handles, fused = [], []
    for i, blk in enumerate(model.blocks):
        fused.append(getattr(blk.attn, "fused_attn", False))
        blk.attn.fused_attn = False

        def hook(_m, inp, _out, i=i):
            a = inp[0]
            attn[i] = a
            if keep_grad and a.requires_grad:
                a.register_hook(lambda g, i=i: grads.__setitem__(i, g))

        handles.append(blk.attn.attn_drop.register_forward_hook(hook))
    try:
        yield attn, grads
    finally:
        for h in handles:
            h.remove()
        for blk, f in zip(model.blocks, fused):
            blk.attn.fused_attn = f


@torch.no_grad()
def attention_maps(model, x) -> torch.Tensor:
    """Returns (blocks, heads, T, T) for a single image."""
    model.eval()
    with capture_attention(model) as (attn, _):
        model(x)
    return torch.stack([attn[i][0] for i in sorted(attn)])


def rollout(attns: List[torch.Tensor], discard_ratio: float = 0.0) -> torch.Tensor:
    """attns: list of (T, T) head-fused matrices. Returns (T, T)."""
    T = attns[0].shape[-1]
    R = torch.eye(T)
    for a in attns:
        a = a.clone().float().cpu()
        if discard_ratio > 0:
            flat = a.view(-1)
            k = int(flat.numel() * discard_ratio)
            if k:
                idx = flat.topk(k, largest=False).indices
                idx = idx[idx != 0]
                flat[idx] = 0
        a = 0.5 * a + 0.5 * torch.eye(T)
        a = a / a.sum(-1, keepdim=True)
        R = a @ R
    return R


def attention_rollout(model, x, head_fusion: str = "mean", discard_ratio: float = 0.0) -> torch.Tensor:
    A = attention_maps(model, x)  # (L, H, T, T)
    fused = A.mean(1) if head_fusion == "mean" else A.max(1).values
    R = rollout(list(fused), discard_ratio)
    prefix = getattr(model, "num_prefix_tokens", 1)
    v = R[0, prefix:]
    g = int(round(v.numel() ** 0.5))
    return v.reshape(g, g)


def grad_rollout(model, x, target_class: int) -> torch.Tensor:
    model.eval()
    with torch.enable_grad():
        with capture_attention(model, keep_grad=True) as (attn, grads):
            xx = x if x.requires_grad else x.detach().clone().requires_grad_(True)
            logits = model(xx)
            logits[:, target_class].sum().backward()
    mats = []
    for i in sorted(attn):
        a, g = attn[i][0].detach(), grads[i][0].detach()
        mats.append(torch.relu(g * a).mean(0))
    R = rollout(mats)
    prefix = getattr(model, "num_prefix_tokens", 1)
    v = R[0, prefix:]
    g = int(round(v.numel() ** 0.5))
    return v.reshape(g, g)
