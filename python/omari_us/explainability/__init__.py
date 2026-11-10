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


@register_method("chefer", ("vit_b16", "usf_mae"))
def _chefer(model, x, target, layer=None):
    return vitx.grad_rollout(model, x, target)


def _metacampp_placeholder(*_a, **_k):
    raise NotImplementedError(
        "MetaCAM++ is reserved for the presenter's research implementation and is not included. "
        "Register one with @register_method('metacampp', models=(...))."
    )


REGISTRY["metacampp"] = Method("metacampp", ("densenet161", "vit_b16", "usf_mae"), _metacampp_placeholder, reserved=True)


def compatible(method: str, model_id: str) -> bool:
    m = REGISTRY.get(method)
    return bool(m and model_id in m.models and not m.reserved)


def methods_for(model_id: str, include_reserved: bool = False):
    return [k for k, m in REGISTRY.items() if model_id in m.models and (include_reserved or not m.reserved)]


def layers_for(method: str, model_id: str) -> Tuple[Optional[str], ...]:
    m = REGISTRY[method]
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
