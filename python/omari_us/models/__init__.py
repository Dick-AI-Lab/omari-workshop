"""Model registry: one place that knows how each of the three workshop models is built and initialised.

  densenet161  torchvision DenseNet-161, DenseNet161_Weights.IMAGENET1K_V1 (supervised ImageNet-1k)
  vit_b16      timm vit_base_patch16_224.augreg_in1k (supervised ImageNet-1k, AugReg; Steiner et al. 2022)
  usf_mae      timm ViT-B/16 initialised from the USF-MAE encoder checkpoint (self-supervised, ultrasound)

`build_model(..., pretrained=False)` gives random initialisation (tests, smoke runs). Fine-tuned checkpoints
saved by omari_us.training are loaded with `load_finetuned`.
"""
from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..utils import NUM_CLASSES

VIT_TIMM_NAME = "vit_base_patch16_224.augreg_in1k"


class DenseNet161Classifier(nn.Module):
    """torchvision DenseNet-161 with an explicit, non-inplace final ReLU (`self.act`).

    torchvision applies `F.relu(features, inplace=True)` inside forward(), which breaks activation/gradient hooks.
    Keys are identical to torchvision (features.*, classifier.*), so official weights load directly.
    The CAM target layer is `self.act` (2208 × 7 × 7 for 224² inputs).
    """

    def __init__(self, num_classes: int = NUM_CLASSES, pretrained: bool = True):
        super().__init__()
        from torchvision.models import DenseNet161_Weights, densenet161

        base = densenet161(weights=DenseNet161_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = base.features
        self.act = nn.ReLU(inplace=False)
        self.classifier = nn.Linear(base.classifier.in_features, num_classes)
        self.init_source = "torchvision DenseNet161_Weights.IMAGENET1K_V1" if pretrained else "random"

    def forward_features(self, x):
        return self.act(self.features(x))

    def forward(self, x):
        a = self.forward_features(x)
        return self.classifier(torch.flatten(F.adaptive_avg_pool2d(a, 1), 1))


def build_densenet161(num_classes: int = NUM_CLASSES, pretrained: bool = True) -> nn.Module:
    return DenseNet161Classifier(num_classes, pretrained)


def build_vit_b16(num_classes: int = NUM_CLASSES, pretrained: bool = True, timm_name: str = VIT_TIMM_NAME, drop_path_rate: float = 0.1) -> nn.Module:
    import timm

    m = timm.create_model(timm_name if pretrained else "vit_base_patch16_224", pretrained=pretrained, num_classes=num_classes, drop_path_rate=drop_path_rate)
    m.init_source = f"timm {timm_name}" if pretrained else "random"
    return m


def build_usf_mae(num_classes: int = NUM_CLASSES, checkpoint: Optional[str] = None, global_pool: str = "token", drop_path_rate: float = 0.1):
    """Requires the USF-MAE checkpoint unless used for tests (checkpoint=None → random init, clearly flagged)."""
    from .usf_mae import load_usf_mae_checkpoint, mae_to_vit_classifier

    if checkpoint is None:
        import timm

        m = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=num_classes, global_pool=global_pool)
        m.init_source = "random (NO USF-MAE checkpoint supplied)"
        return m
    sd, rep = load_usf_mae_checkpoint(checkpoint)
    m = mae_to_vit_classifier(sd, num_classes=num_classes, global_pool=global_pool, drop_path_rate=drop_path_rate)
    m.init_source = f"USF-MAE encoder {os.path.basename(checkpoint)} sha256:{rep.sha256[:12]}"
    m.checkpoint_report = rep
    return m


def build_model(model_id: str, num_classes: int = NUM_CLASSES, pretrained: bool = True, usf_mae_checkpoint: Optional[str] = None, **kw) -> nn.Module:
    if model_id == "densenet161":
        return build_densenet161(num_classes, pretrained)
    if model_id == "vit_b16":
        return build_vit_b16(num_classes, pretrained, **kw)
    if model_id == "usf_mae":
        return build_usf_mae(num_classes, usf_mae_checkpoint if pretrained else None, **kw)
    raise ValueError(f"unknown model {model_id}")


def classifier_param_names(model: nn.Module, model_id: str) -> Tuple[str, ...]:
    """Parameters trained in the frozen-encoder ("linear probe") regime."""
    if model_id == "densenet161":
        return ("classifier.",)
    return ("head.", "fc_norm.")


def save_finetuned(model: nn.Module, path: str, meta: Dict) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "meta": meta}, path)


def load_finetuned(model_id: str, path: str, map_location="cpu") -> Tuple[nn.Module, Dict]:
    """Rebuild the architecture (random init) and load fine-tuned weights saved by `save_finetuned`."""
    obj = torch.load(path, map_location=map_location, weights_only=False)
    meta = obj.get("meta", {})
    kw = {}
    if model_id == "usf_mae":
        kw["global_pool"] = meta.get("global_pool", "token")
    model = build_model(model_id, meta.get("num_classes", NUM_CLASSES), pretrained=False, **kw)
    model.load_state_dict(obj["state_dict"], strict=True)
    model.init_source = meta.get("init_source", "unknown")
    return model.eval(), meta
