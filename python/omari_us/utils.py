"""Shared helpers: seeding, hashing, provenance, devices, YAML configs."""
from __future__ import annotations

import datetime as _dt
import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

CLASSES = ["Fetal abdomen", "Fetal brain", "Fetal femur", "Fetal thorax", "Maternal cervix", "Other"]
CLASS_SHORT = ["Abdomen", "Brain", "Femur", "Thorax", "Cervix", "Other"]
NUM_CLASSES = len(CLASSES)
MODEL_IDS = ("densenet161", "vit_b16", "usf_mae")


def seed_everything(seed: int = 42, deterministic: bool = False) -> None:
    """Seed Python, NumPy and PyTorch (CPU + CUDA). `deterministic=True` also forces deterministic kernels."""
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:
        import torch

        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        if deterministic:
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            torch.use_deterministic_algorithms(True, warn_only=True)
    except ImportError:  # pragma: no cover
        pass


def get_device(prefer: Optional[str] = None):
    import torch

    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def sha256_file(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def git_commit(cwd: Optional[str] = None) -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True, text=True, timeout=5)
        return out.stdout.strip() or None
    except Exception:
        return None


def now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


def provenance(script: str, checkpoint_id: Optional[str] = None, checkpoint_sha256: Optional[str] = None,
               params: Optional[Dict[str, Any]] = None, note: Optional[str] = None) -> Dict[str, Any]:
    """Provenance record attached to every exported artifact (mirrors `Provenance` in website/src/data/types.ts)."""
    p: Dict[str, Any] = {"script": script, "generated_at": now_iso(), "git_commit": git_commit()}
    if checkpoint_id:
        p["checkpoint_id"] = checkpoint_id
    if checkpoint_sha256:
        p["checkpoint_sha256"] = checkpoint_sha256
    if params:
        p["params"] = params
    if note:
        p["note"] = note
    return {k: v for k, v in p.items() if v is not None}


def load_yaml(path: str | os.PathLike) -> Dict[str, Any]:
    import yaml

    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    # allow environment-variable expansion in string values, e.g. ${DATA_ROOT}
    def _expand(v):
        if isinstance(v, str):
            return os.path.expandvars(os.path.expanduser(v))
        if isinstance(v, dict):
            return {k: _expand(x) for k, x in v.items()}
        if isinstance(v, list):
            return [_expand(x) for x in v]
        return v

    return _expand(cfg)


def write_json(obj: Any, path: str | os.PathLike, indent: Optional[int] = None) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)

    def _default(o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        raise TypeError(f"not JSON serialisable: {type(o)}")

    with open(path, "w") as f:
        json.dump(obj, f, indent=indent, default=_default)
