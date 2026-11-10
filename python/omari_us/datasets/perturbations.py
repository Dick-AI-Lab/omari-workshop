"""Controlled image perturbations. Semantics mirror website/src/lib/imageOps.ts exactly, so that a precomputed
perturbation and the browser slider mean the same thing. Inputs/outputs: float32 arrays in [0, 1], shape (H, W).
"""
from __future__ import annotations

from typing import Dict, Optional, Tuple

import numpy as np


def brightness(img: np.ndarray, delta: float) -> np.ndarray:
    return np.clip(img + delta, 0, 1).astype(np.float32)


def contrast(img: np.ndarray, gain: float) -> np.ndarray:
    """Gain about mid-grey 0.5 (website: (v − 0.5)·c + 0.5)."""
    return np.clip((img - 0.5) * gain + 0.5, 0, 1).astype(np.float32)


def gamma(img: np.ndarray, g: float) -> np.ndarray:
    return np.power(np.clip(img, 0, 1), g).astype(np.float32)


def resolution(img: np.ndarray, factor: int) -> np.ndarray:
    """Area-downsample by `factor`, then nearest-upsample back (website default)."""
    from PIL import Image

    if factor <= 1:
        return img.copy()
    h, w = img.shape
    pil = Image.fromarray((img * 255).round().astype(np.uint8))
    small = pil.resize((max(2, round(w / factor)), max(2, round(h / factor))), Image.BOX)
    return (np.asarray(small.resize((w, h), Image.NEAREST), dtype=np.float32) / 255.0)


def sector_mask(img: np.ndarray, thresh: float = 0.02) -> np.ndarray:
    return img > thresh


def speckle(img: np.ndarray, strength: float, looks: int = 1, seed: int = 0, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """I·(1 + s·(n − 1)), n ~ Gamma(L, 1/L); applied inside the imaging sector only."""
    rng = np.random.default_rng(seed)
    n = rng.gamma(shape=looks, scale=1.0 / looks, size=img.shape)
    out = img * (1 + strength * (n - 1))
    m = sector_mask(img) if mask is None else mask
    out = np.where(m, out, img)
    return np.clip(out, 0, 1).astype(np.float32)


def annotation_mask(img: np.ndarray, thresh: float = 0.92, dilate: int = 1) -> np.ndarray:
    """Bright, near-saturated pixels (burned-in text, calipers, markers). Crude by design; inspect before trusting."""
    m = img >= thresh
    if dilate > 0:
        from scipy.ndimage import binary_dilation

        m = binary_dilation(m, iterations=dilate)
    return m


def inpaint(img: np.ndarray, mask: np.ndarray, radius: int = 3) -> np.ndarray:
    """Telea inpainting if OpenCV is installed (same family as the USF-MAE preprocessing notebook), else local-median fill."""
    try:
        import cv2

        u8 = (img * 255).round().astype(np.uint8)
        out = cv2.inpaint(u8, mask.astype(np.uint8) * 255, radius, cv2.INPAINT_TELEA)
        return out.astype(np.float32) / 255.0
    except ImportError:
        from scipy.ndimage import median_filter

        med = median_filter(np.where(mask, 0, img), size=2 * radius + 3)
        return np.where(mask, med, img).astype(np.float32)


def occlude(img: np.ndarray, region: np.ndarray, mode: str = "blur", sigma: float = 8.0) -> np.ndarray:
    """Replace `region` (bool mask) with a blurred copy (default; closer to the data distribution) or mean grey."""
    if mode == "mean":
        return np.where(region, img[~region].mean() if (~region).any() else 0.5, img).astype(np.float32)
    from scipy.ndimage import gaussian_filter

    return np.where(region, gaussian_filter(img, sigma), img).astype(np.float32)


DEFAULT_GRID: Tuple[Tuple[str, float, str], ...] = (
    ("contrast", 0.6, "contrast ×0.6"),
    ("contrast", 1.5, "contrast ×1.5"),
    ("brightness", -0.15, "brightness −0.15"),
    ("brightness", 0.15, "brightness +0.15"),
    ("speckle", 0.3, "speckle 0.3 (L=1)"),
    ("speckle", 0.6, "speckle 0.6 (L=1)"),
    ("resolution", 2, "resolution ½"),
    ("resolution", 4, "resolution ¼"),
    ("occlude_annotations", 0.92, "overlays inpainted"),
)


def apply(img: np.ndarray, kind: str, param: float, seed: int = 0, region: Optional[np.ndarray] = None) -> np.ndarray:
    if kind == "brightness":
        return brightness(img, param)
    if kind == "contrast":
        return contrast(img, param)
    if kind == "gamma":
        return gamma(img, param)
    if kind == "resolution":
        return resolution(img, int(param))
    if kind == "speckle":
        return speckle(img, param, looks=1, seed=seed)
    if kind == "occlude_annotations":
        return inpaint(img, annotation_mask(img, param))
    if kind == "occlude_patch":
        if region is None:
            raise ValueError("occlude_patch needs a region mask")
        return occlude(img, region)
    if kind == "flip":
        return img[:, ::-1].copy()
    raise ValueError(kind)
