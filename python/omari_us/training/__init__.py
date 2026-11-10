"""Fine-tuning loop shared by notebooks 02–04 and scripts/train.py.

Regimes
  frozen    linear probe: only the classification head (and fc_norm if present) is trained
  finetune  all parameters; ViTs use layer-wise learning-rate decay (MAE/BEiT recipe), DenseNet a lower
            backbone learning rate (backbone_lr_mult)

Model selection uses validation macro-F1 on a patient-disjoint validation set. The test split is touched once,
after training, to write test_predictions.npz. Every run writes config.json, history.json, best.pt and provenance.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..datasets import FetalPlanesDataset, build_transforms, class_weights, input_spec, label_fraction_subset
from ..evaluation.metrics import classification_report
from ..models import build_model, classifier_param_names, save_finetuned
from ..utils import NUM_CLASSES, get_device, git_commit, now_iso, seed_everything, sha256_file, write_json


@dataclass
class TrainConfig:
    model: str = "densenet161"
    regime: str = "finetune"  # frozen | finetune
    epochs: int = 20
    batch_size: int = 32
    lr: float = 1e-4
    weight_decay: float = 0.05
    warmup_epochs: int = 1
    layer_decay: float = 0.75  # ViTs only (finetune)
    backbone_lr_mult: float = 0.1  # DenseNet only (finetune)
    loss: str = "weighted_ce"  # ce | weighted_ce | focal
    weight_scheme: str = "inverse"
    label_smoothing: float = 0.1
    focal_gamma: float = 2.0
    label_fraction: float = 1.0
    seed: int = 0
    num_workers: int = 2
    amp: bool = True
    patience: int = 6
    usf_mae_checkpoint: Optional[str] = None
    usf_mae_global_pool: str = "token"
    input_overrides: Dict = field(default_factory=dict)
    augment: Dict = field(default_factory=dict)
    max_train_batches: Optional[int] = None  # demo mode
    max_eval_batches: Optional[int] = None
    out_dir: str = "runs"
    run_name: Optional[str] = None
    pretrained: bool = True  # False only for smoke tests (random init, no downloads)

    def name(self) -> str:
        return self.run_name or f"{self.model}_{self.regime}_lf{self.label_fraction:g}_s{self.seed}"


def focal_loss(logits, target, weight=None, gamma=2.0):
    logp = F.log_softmax(logits, dim=-1)
    pt = logp.gather(1, target[:, None]).squeeze(1).exp()
    loss = -((1 - pt) ** gamma) * logp.gather(1, target[:, None]).squeeze(1)
    if weight is not None:
        loss = loss * weight[target]
        return loss.sum() / weight[target].sum()
    return loss.mean()


def make_criterion(cfg: TrainConfig, train_labels: np.ndarray, device):
    w = torch.tensor(class_weights(train_labels, cfg.weight_scheme), dtype=torch.float32, device=device) if cfg.loss != "ce" else None
    if cfg.loss == "focal":
        return lambda lo, y: focal_loss(lo, y, w, cfg.focal_gamma), w
    return nn.CrossEntropyLoss(weight=w, label_smoothing=cfg.label_smoothing), w


def param_groups(model: nn.Module, cfg: TrainConfig) -> List[Dict]:
    head = classifier_param_names(model, cfg.model)
    if cfg.regime == "frozen":
        for n, p in model.named_parameters():
            p.requires_grad = n.startswith(head)
        return [{"params": [p for p in model.parameters() if p.requires_grad], "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    for p in model.parameters():
        p.requires_grad = True
    if cfg.model == "densenet161":
        hp = [p for n, p in model.named_parameters() if n.startswith(head)]
        bp = [p for n, p in model.named_parameters() if not n.startswith(head)]
        return [{"params": bp, "lr": cfg.lr * cfg.backbone_lr_mult, "weight_decay": cfg.weight_decay},
                {"params": hp, "lr": cfg.lr, "weight_decay": cfg.weight_decay}]
    # ViT: layer-wise lr decay, no weight decay on norms/biases/tokens
    depth = len(model.blocks)
    groups: Dict[str, Dict] = {}
    for n, p in model.named_parameters():
        if n.startswith(("cls_token", "pos_embed", "patch_embed")):
            layer = 0
        elif n.startswith("blocks."):
            layer = int(n.split(".")[1]) + 1
        else:
            layer = depth + 1
        no_wd = p.ndim == 1 or n in ("cls_token", "pos_embed")
        key = f"l{layer}_{'nowd' if no_wd else 'wd'}"
        scale = cfg.layer_decay ** (depth + 1 - layer)
        g = groups.setdefault(key, {"params": [], "lr": cfg.lr * scale, "weight_decay": 0.0 if no_wd else cfg.weight_decay, "lr_scale": scale})
        g["params"].append(p)
    return list(groups.values())


def cosine_with_warmup(optimizer, warmup_steps: int, total_steps: int):
    base = [g["lr"] for g in optimizer.param_groups]

    def set_lr(step: int):
        if step < warmup_steps:
            f = (step + 1) / max(1, warmup_steps)
        else:
            t = (step - warmup_steps) / max(1, total_steps - warmup_steps)
            f = 0.5 * (1 + math.cos(math.pi * min(1.0, t)))
        for g, b in zip(optimizer.param_groups, base):
            g["lr"] = b * f
        return f

    return set_lr


@torch.no_grad()
def predict(model: nn.Module, loader, device, amp: bool = False, max_batches: Optional[int] = None):
    model.eval()
    probs, labels, idx = [], [], []
    for b, (x, y, i) in enumerate(loader):
        if max_batches is not None and b >= max_batches:
            break
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            lo = model(x.to(device))
        probs.append(F.softmax(lo.float(), dim=-1).cpu().numpy())
        labels.append(y.numpy())
        idx.append(i.numpy())
    return np.concatenate(probs), np.concatenate(labels), np.concatenate(idx)


def train(cfg: TrainConfig, splits: pd.DataFrame, progress: bool = True) -> Dict:
    """Train one run. `splits` is a metadata frame with a `split` column ('train' | 'val' | 'test')."""
    from torch.utils.data import DataLoader

    seed_everything(cfg.seed)
    device = get_device()
    run_dir = os.path.join(cfg.out_dir, cfg.name())
    os.makedirs(run_dir, exist_ok=True)
    spec = input_spec(cfg.model, cfg.input_overrides)
    tr_df = label_fraction_subset(splits, cfg.label_fraction, seed=cfg.seed, split="train")
    va_df = splits[splits["split"] == "val"]
    te_df = splits[splits["split"] == "test"]
    kw = dict(num_workers=cfg.num_workers, pin_memory=device.type == "cuda")
    tl = DataLoader(FetalPlanesDataset(tr_df, build_transforms(spec, True, cfg.augment)), batch_size=cfg.batch_size, shuffle=True, drop_last=len(tr_df) > cfg.batch_size, **kw)
    vl = DataLoader(FetalPlanesDataset(va_df, build_transforms(spec, False)), batch_size=cfg.batch_size * 2, **kw)
    sl = DataLoader(FetalPlanesDataset(te_df, build_transforms(spec, False)), batch_size=cfg.batch_size * 2, **kw)

    model = build_model(cfg.model, NUM_CLASSES, pretrained=cfg.pretrained or cfg.model == "usf_mae", usf_mae_checkpoint=cfg.usf_mae_checkpoint,
                        **({"global_pool": cfg.usf_mae_global_pool} if cfg.model == "usf_mae" else {})).to(device)
    init_source = getattr(model, "init_source", "unknown")
    if cfg.model == "usf_mae" and cfg.usf_mae_checkpoint is None:
        raise ValueError("usf_mae requires usf_mae_checkpoint (path to the official 100-epoch checkpoint)")
    groups = param_groups(model, cfg)
    opt = torch.optim.AdamW(groups, lr=cfg.lr, weight_decay=cfg.weight_decay)
    steps_per_epoch = min(len(tl), cfg.max_train_batches or len(tl))
    set_lr = cosine_with_warmup(opt, cfg.warmup_epochs * steps_per_epoch, cfg.epochs * steps_per_epoch)
    crit, w = make_criterion(cfg, tr_df["label"].to_numpy(), device)
    scaler = torch.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")

    history = {"epoch": [], "train_loss": [], "val_loss": [], "val_macro_f1": [], "val_accuracy": [], "lr": [], "seconds": []}
    best, best_epoch, bad, step = -1.0, -1, 0, 0
    iterator = range(cfg.epochs)
    if progress:
        try:
            from tqdm.auto import tqdm

            iterator = tqdm(iterator, desc=cfg.name())
        except ImportError:
            pass
    for epoch in iterator:
        t0 = time.time()
        model.train()
        tot, n = 0.0, 0
        for b, (x, y, _) in enumerate(tl):
            if cfg.max_train_batches is not None and b >= cfg.max_train_batches:
                break
            set_lr(step)
            step += 1
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=cfg.amp and device.type == "cuda"):
                loss = crit(model(x), y)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            tot += float(loss) * len(y)
            n += len(y)
        vp, vy, _ = predict(model, vl, device, cfg.amp, cfg.max_eval_batches)
        vloss = float(F.nll_loss(torch.log(torch.tensor(vp).clamp_min(1e-12)), torch.tensor(vy), weight=w.cpu() if w is not None else None))
        rep = classification_report(vy, vp.argmax(1))
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(tot / max(1, n))
        history["val_loss"].append(vloss)
        history["val_macro_f1"].append(rep["macro_f1"])
        history["val_accuracy"].append(rep["accuracy"])
        history["lr"].append(opt.param_groups[-1]["lr"])
        history["seconds"].append(time.time() - t0)
        write_json(history, os.path.join(run_dir, "history.json"))
        if rep["macro_f1"] > best:
            best, best_epoch, bad = rep["macro_f1"], epoch + 1, 0
            meta = {"model": cfg.model, "num_classes": NUM_CLASSES, "init_source": init_source, "config": asdict(cfg),
                    "epoch": epoch + 1, "val_macro_f1": best, "global_pool": cfg.usf_mae_global_pool, "input": spec.to_dict(),
                    "created": now_iso(), "git_commit": git_commit()}
            save_finetuned(model, os.path.join(run_dir, "best.pt"), meta)
        else:
            bad += 1
            if bad >= cfg.patience:
                break

    # final test pass with the best checkpoint
    from ..models import load_finetuned

    best_model, meta = load_finetuned(cfg.model, os.path.join(run_dir, "best.pt"))
    best_model.to(device)
    tp, ty, ti = predict(best_model, sl, device, cfg.amp, cfg.max_eval_batches)
    vp, vy, vi = predict(best_model, vl, device, cfg.amp, cfg.max_eval_batches)
    tdf = te_df.reset_index(drop=True).iloc[ti]
    vdf = va_df.reset_index(drop=True).iloc[vi]
    np.savez_compressed(os.path.join(run_dir, "test_predictions.npz"), probs=tp, labels=ty, ids=tdf["Image_name"].to_numpy(),
                        patients=tdf["Patient_num"].to_numpy(), machines=tdf.get("US_Machine", pd.Series([""] * len(tdf))).to_numpy())
    np.savez_compressed(os.path.join(run_dir, "val_predictions.npz"), probs=vp, labels=vy, ids=vdf["Image_name"].to_numpy(), patients=vdf["Patient_num"].to_numpy())
    ckpt_sha = sha256_file(os.path.join(run_dir, "best.pt"))
    summary = {"run": cfg.name(), "best_epoch": best_epoch, "best_val_macro_f1": best, "test": classification_report(ty, tp.argmax(1)),
               "n_train_images": int(len(tr_df)), "n_train_patients": int(tr_df["Patient_num"].nunique()), "init_source": init_source,
               "checkpoint_sha256": ckpt_sha, "config": asdict(cfg), "input": spec.to_dict(), "finished": now_iso(), "git_commit": git_commit(),
               "demo_mode": cfg.max_train_batches is not None}
    write_json(summary, os.path.join(run_dir, "summary.json"), indent=2)
    return summary
