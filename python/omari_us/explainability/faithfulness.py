"""Faithfulness and robustness checks for attribution maps.

  deletion / insertion curves (Petsiuk et al., 2018): progressively remove (or reveal) pixels in order of attribution
  cascading randomisation sanity check (Adebayo et al., 2018)
  stability: rank correlation of maps under small input perturbations
"""
from __future__ import annotations

import copy
from typing import Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def _order(attr: torch.Tensor, size) -> torch.Tensor:
    up = F.interpolate(attr[None, None].float(), size=size, mode="bilinear", align_corners=False)[0, 0]
    return torch.argsort(up.flatten(), descending=True)


@torch.no_grad()
def deletion_insertion(model: nn.Module, x: torch.Tensor, attr: torch.Tensor, target_class: int, steps: int = 20,
                       baseline: Optional[torch.Tensor] = None, mode: str = "deletion") -> Dict[str, object]:
    """Probability of `target_class` as the top-attributed pixels are replaced by `baseline` (deletion) or revealed from it (insertion).

    baseline defaults to a heavy Gaussian blur of x (keeps the image closer to the data distribution than black).
    Returns dict(curve=list, auc=float). Lower deletion AUC / higher insertion AUC = more faithful.
    """
    model.eval()
    H, W = x.shape[-2:]
    if baseline is None:
        from torchvision.transforms.functional import gaussian_blur

        baseline = gaussian_blur(x, kernel_size=51, sigma=15.0)
    order = _order(attr, (H, W))
    n = H * W
    cur = x.clone() if mode == "deletion" else baseline.clone()
    src = baseline if mode == "deletion" else x
    curve = [float(F.softmax(model(cur), -1)[0, target_class])]
    flat_cur = cur.view(cur.shape[0], cur.shape[1], -1)
    flat_src = src.view(src.shape[0], src.shape[1], -1)
    for s in range(1, steps + 1):
        idx = order[int((s - 1) / steps * n): int(s / steps * n)]
        flat_cur[:, :, idx] = flat_src[:, :, idx]
        curve.append(float(F.softmax(model(cur), -1)[0, target_class]))
    auc = float(np.trapezoid(curve, dx=1.0 / steps)) if hasattr(np, "trapezoid") else float(np.trapz(curve, dx=1.0 / steps))
    return {"curve": curve, "auc": auc, "mode": mode}


def spearman(a: torch.Tensor, b: torch.Tensor) -> float:
    from scipy.stats import spearmanr

    if a.shape != b.shape:
        b = F.interpolate(b[None, None].float(), size=a.shape, mode="bilinear", align_corners=False)[0, 0]
    return float(spearmanr(a.flatten().cpu().numpy(), b.flatten().cpu().numpy()).statistic)


def cascading_randomisation(model: nn.Module, x: torch.Tensor, target_class: int, explain: Callable[[nn.Module], torch.Tensor],
                            layer_groups: Sequence[str], seed: int = 0) -> List[Dict[str, float]]:
    """Re-initialise parameter groups from the top down (names by prefix) and compare maps with the original.

    `explain(model)` must return a map for x / target_class. High similarity after randomisation is a red flag.
    """
    torch.manual_seed(seed)
    ref = explain(model).detach()
    m = copy.deepcopy(model)
    for p in m.parameters():
        p.requires_grad_(False)  # explanations only need activation gradients
    out = []
    for prefix in layer_groups:
        for n, p in m.named_parameters():
            if n.startswith(prefix):
                with torch.no_grad():
                    if p.ndim >= 2:
                        nn.init.kaiming_normal_(p) if p.ndim == 4 else nn.init.trunc_normal_(p, std=0.02)
                    else:
                        nn.init.normal_(p, std=0.02)
        cur = explain(m).detach()
        out.append({"randomised_through": prefix, "spearman": spearman(ref, cur)})
    del m
    return out
