"""Model-specific input transforms. Each model keeps the preprocessing its pretrained weights expect.

Defaults (overridable in configs/*.yaml under `input:`):
  densenet161  224², stretch, bilinear, ImageNet mean/std              (torchvision IMAGENET1K_V1 statistics)
  vit_b16      224², stretch, bicubic,  mean = std = 0.5               (timm vit_base_patch16_224.augreg_in1k cfg)
  usf_mae      224², stretch, nearest,  ImageNet mean/std              (official USF-MAE US_data_loading.py)

All three use the same geometry ('stretch' to 224×224) so that spatial maps are directly comparable; the
website's preprocessing lab shows the cost of that choice. Grey images are replicated to 3 channels.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


@dataclass
class InputSpec:
    size: int = 224
    resize: str = "stretch"  # stretch | letterbox | centercrop
    interpolation: str = "bilinear"  # nearest | bilinear | bicubic
    mean: List[float] = field(default_factory=lambda: list(IMAGENET_MEAN))
    std: List[float] = field(default_factory=lambda: list(IMAGENET_STD))
    channels: int = 3

    def to_dict(self) -> Dict:
        return {"size": self.size, "resize": self.resize, "interpolation": self.interpolation, "mean": self.mean, "std": self.std, "channels": self.channels}


DEFAULT_INPUT: Dict[str, InputSpec] = {
    "densenet161": InputSpec(interpolation="bilinear"),
    "vit_b16": InputSpec(interpolation="bicubic", mean=[0.5] * 3, std=[0.5] * 3),
    "usf_mae": InputSpec(interpolation="nearest"),
}


def input_spec(model_id: str, overrides: Optional[Dict] = None) -> InputSpec:
    base = DEFAULT_INPUT[model_id]
    d = base.to_dict()
    d.update(overrides or {})
    return InputSpec(**d)


class Letterbox:
    """Resize preserving aspect ratio and zero-pad to a square."""

    def __init__(self, size: int, interpolation):
        self.size, self.interpolation = size, interpolation

    def __call__(self, img):
        from PIL import Image

        w, h = img.size
        s = self.size / max(w, h)
        nw, nh = max(1, round(w * s)), max(1, round(h * s))
        r = img.resize((nw, nh), self.interpolation)
        out = Image.new(img.mode, (self.size, self.size), 0)
        out.paste(r, ((self.size - nw) // 2, (self.size - nh) // 2))
        return out


def _interp(name: str):
    from torchvision.transforms import InterpolationMode

    return {"nearest": InterpolationMode.NEAREST, "bilinear": InterpolationMode.BILINEAR, "bicubic": InterpolationMode.BICUBIC}[name]


def _pil_interp(name: str):
    from PIL import Image

    return {"nearest": Image.NEAREST, "bilinear": Image.BILINEAR, "bicubic": Image.BICUBIC}[name]


def geometry(spec: InputSpec):
    from torchvision import transforms as T

    if spec.resize == "stretch":
        return T.Resize((spec.size, spec.size), interpolation=_interp(spec.interpolation))
    if spec.resize == "letterbox":
        return Letterbox(spec.size, _pil_interp(spec.interpolation))
    if spec.resize == "centercrop":
        return T.Compose([T.Resize(spec.size, interpolation=_interp(spec.interpolation)), T.CenterCrop(spec.size)])
    raise ValueError(spec.resize)


def build_transforms(spec: InputSpec, train: bool, augment: Optional[Dict] = None):
    """Eval: grey→RGB, geometry, ToTensor, Normalize. Train adds mild, ultrasound-plausible augmentation.

    augment keys (defaults): rotation=10 (deg), scale=(0.85, 1.0), hflip=0.5, brightness=0.15, contrast=0.15.
    Vertical flips are off by default: the probe is always at the top of the sector.
    """
    from torchvision import transforms as T

    ops = [T.Grayscale(num_output_channels=3)]
    if train:
        a = {"rotation": 10, "scale": (0.85, 1.0), "hflip": 0.5, "brightness": 0.15, "contrast": 0.15}
        a.update(augment or {})
        ops += [geometry(spec)]
        if a["rotation"] or a["scale"] != (1.0, 1.0):
            ops += [T.RandomAffine(degrees=a["rotation"], scale=tuple(a["scale"]), interpolation=_interp("bilinear"))]
        if a["hflip"]:
            ops += [T.RandomHorizontalFlip(a["hflip"])]
        if a["brightness"] or a["contrast"]:
            ops += [T.ColorJitter(brightness=a["brightness"], contrast=a["contrast"])]
    else:
        ops += [geometry(spec)]
    ops += [T.ToTensor(), T.Normalize(spec.mean, spec.std)]
    return T.Compose(ops)


def denormalize(x, spec: InputSpec):
    """Tensor (B,3,H,W) or (3,H,W) normalised → [0,1] grey (mean over channels)."""
    import torch

    mean = torch.tensor(spec.mean, device=x.device).view(-1, 1, 1)
    std = torch.tensor(spec.std, device=x.device).view(-1, 1, 1)
    y = x * std + mean
    return y.mean(dim=-3).clamp(0, 1)
