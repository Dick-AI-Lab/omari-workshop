"""Attribution method registry with explicit architecture compatibility.

Every method is registered with the models it supports. `explain(method, model_id, model, x, target, layer)`
dispatches and raises `IncompatibleMethod` for unsupported pairs instead of silently producing a map.

MetaCAM++ has a reserved slot: it is *not implemented here*. Supply an implementation with
`@register_method("metacampp", models=(...))` and the exporter / website pick it up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

import torch

from . import cam as cam  # noqa: F401
from . import faithfulness as faithfulness  # noqa: F401
from . import vit as vitx
from .cam import cam_densenet, gradcam, gradcampp, normalise, upsample  # noqa: F401


class IncompatibleMethod(ValueError):
    pass


@dataclass
class Method:
    name: str
    models: Tuple[str, ...]
    fn: Callable
    class_specific: bool = True
    layers: Optional[Dict[str, Tuple[str, ...]]] = None  # model_id → selectable layers
    reserved: bool = False


REGISTRY: Dict[str, Method] = {}


def register_method(name: str, models: Tuple[str, ...], class_specific: bool = True, layers: Optional[Dict[str, Tuple[str, ...]]] = None):
    def deco(fn):
        REGISTRY[name] = Method(name, tuple(models), fn, class_specific, layers, reserved=False)
        return fn

    return deco


DENSENET_LAYERS = ("act", "features.denseblock3", "features.denseblock2")
VIT_BLOCKS = ("block11", "block8", "block5")


def _densenet_layer(model, layer: Optional[str]):
    if layer in (None, "act", "final"):
        return model.act
    mod = model
    for part in layer.split("."):
        mod = getattr(mod, part)
    return mod


def _vit_block(layer: Optional[str]) -> int:
    if layer in (None, "final"):
        return -1
    return int(str(layer).replace("block", ""))


@register_method("cam", ("densenet161",), layers={"densenet161": ("act",)})
def _cam(model, x, target, layer=None):
    if layer not in (None, "act", "final"):
        raise IncompatibleMethod("CAM is only defined at the final GAP-linear features")
    return cam_densenet(model, x, target)


@register_method("gradcam", ("densenet161",), layers={"densenet161": DENSENET_LAYERS})
def _gc(model, x, target, layer=None):
    return gradcam(model, x, target, _densenet_layer(model, layer))


@register_method("gradcampp", ("densenet161",), layers={"densenet161": DENSENET_LAYERS})
def _gcpp(model, x, target, layer=None):
    return gradcampp(model, x, target, _densenet_layer(model, layer))


@register_method("vit_gradcam", ("vit_b16", "usf_mae"), layers={"vit_b16": VIT_BLOCKS, "usf_mae": VIT_BLOCKS})
def _vgc(model, x, target, layer=None):
    return vitx.token_gradcam(model, x, target, _vit_block(layer))


@register_method("vit_gradcampp", ("vit_b16", "usf_mae"), layers={"vit_b16": VIT_BLOCKS, "usf_mae": VIT_BLOCKS})
def _vgcpp(model, x, target, layer=None):
    return vitx.token_gradcampp(model, x, target, _vit_block(layer))


@register_method("attention_rollout", ("vit_b16", "usf_mae"), class_specific=False)
def _roll(model, x, target=None, layer=None):
    return vitx.attention_rollout(model, x)


@register_method("grad_rollout", ("vit_b16", "usf_mae"))
def _grad_rollout(model, x, target, layer=None):
    """Gradient-weighted attention rollout: E_h[ReLU(dy_c/dA * A)] per block, fed through the rollout recursion
    (0.5*A + 0.5*I, row-normalised). A simplified variant *inspired by* Chefer et al. (2021) without LRP and without
    their R <- R + A_bar R update. Bundles written before v1.1 call this method "chefer" (see METHOD_ALIASES)."""
    return vitx.grad_rollout(model, x, target)


# Old method ids still accepted on input (bundles written with schema 1.0 used "chefer").
METHOD_ALIASES: Dict[str, str] = {"chefer": "grad_rollout"}


def canonical_method(method: str) -> str:
    return METHOD_ALIASES.get(method, method)


METHOD_LABELS: Dict[str, str] = {
    "cam": "CAM",
    "gradcam": "Grad-CAM",
    "gradcampp": "Grad-CAM++",
    "vit_gradcam": "Token Grad-CAM",
    "vit_gradcampp": "Token Grad-CAM++",
    "attention_rollout": "Attention rollout",
    "grad_rollout": "Gradient-weighted attention rollout",
    "metacampp": "MetaCAM++ (reserved, not implemented)",
}

N_VIT_BLOCKS = 12


def method_label(method: str) -> str:
    m = canonical_method(method)
    return METHOD_LABELS.get(m, m)


def layer_label(model_id: str, layer: Optional[str], method: Optional[str] = None) -> str:
    """Human-readable layer name. ViT ids `blockN` are the *input* of model.blocks[N] (0-indexed), i.e. the residual
    stream after N transformer blocks; displayed 1-indexed as "after block N of 12" (bundle schema 1.1, decision 4)."""
    method = canonical_method(method) if method else None
    if layer in (None, "final"):
        if method in ("attention_rollout", "grad_rollout"):
            return f"all {N_VIT_BLOCKS} blocks (rollout)"
        if model_id == "densenet161":
            return "final features (ReLU(norm5), 7x7)"
        return f"after block {N_VIT_BLOCKS - 1} of {N_VIT_BLOCKS}" if model_id in ("vit_b16", "usf_mae") else "final"
    if model_id == "densenet161":
        if layer == "act":
            return "final features (ReLU(norm5), 7x7)"
        if layer.startswith("features.denseblock"):
            n = layer.replace("features.denseblock", "")
            res = {"1": 56, "2": 28, "3": 14, "4": 7}.get(n)
            return f"dense block {n} output" + (f" ({res}x{res})" if res else "")
        return layer
    if str(layer).startswith("block"):
        n = int(str(layer).replace("block", ""))
        return f"after block {n} of {N_VIT_BLOCKS}"
    return str(layer)


def _metacampp_placeholder(*_a, **_k):
    raise NotImplementedError(
        "MetaCAM++ is reserved for the presenter's research implementation and is not included. "
        "Register one with @register_method('metacampp', models=(...))."
    )


REGISTRY["metacampp"] = Method("metacampp", ("densenet161", "vit_b16", "usf_mae"), _metacampp_placeholder, reserved=True)


def compatible(method: str, model_id: str) -> bool:
    m = REGISTRY.get(canonical_method(method))
    return bool(m and model_id in m.models and not m.reserved)


def methods_for(model_id: str, include_reserved: bool = False):
    return [k for k, m in REGISTRY.items() if model_id in m.models and (include_reserved or not m.reserved)]


def layers_for(method: str, model_id: str) -> Tuple[Optional[str], ...]:
    m = REGISTRY[canonical_method(method)]
    return (m.layers or {}).get(model_id, (None,))


class no_param_grads:
    """Temporarily set requires_grad=False on all parameters so attribution backward passes only compute
    gradients w.r.t. activations (the input is made to require grad instead). Saves ~4 bytes × #params of memory
    per call and avoids leaving .grad tensors behind."""

    def __init__(self, model):
        self.model, self.flags = model, None

    def __enter__(self):
        self.flags = [p.requires_grad for p in self.model.parameters()]
        for p in self.model.parameters():
            p.requires_grad_(False)
            p.grad = None
        return self.model

    def __exit__(self, *exc):
        for p, f in zip(self.model.parameters(), self.flags):
            p.requires_grad_(f)


def explain(method: str, model_id: str, model, x: torch.Tensor, target: int, layer: Optional[str] = None) -> torch.Tensor:
    method = canonical_method(method)
    m = REGISTRY.get(method)
    if m is None:
        raise KeyError(f"unknown method {method}; available: {sorted(REGISTRY)}")
    if model_id not in m.models:
        raise IncompatibleMethod(f"{method} does not support {model_id} (supports {m.models})")
    if m.reserved:
        return m.fn()
    model.eval()
    with no_param_grads(model):
        return m.fn(model, x.detach().clone().requires_grad_(True), target, layer).detach().float().cpu()
