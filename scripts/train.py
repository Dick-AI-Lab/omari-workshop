#!/usr/bin/env python
"""Fine-tune one model. Example:
    python scripts/train.py --config configs/usf_mae.yaml --regime finetune --splits artifacts/splits/split_manifest.csv
    python scripts/train.py --config configs/densenet161.yaml --regime frozen --label-fraction 0.05 --seed 1
    python scripts/train.py --config configs/vit_b16.yaml --all-fractions   # label-efficiency sweep (fractions × seeds)
Add --demo for a few batches only (pipeline check; results are meaningless).
"""
import argparse

import _path  # noqa: F401
from omari_us.datasets import load_split_manifest
from omari_us.training import TrainConfig, train
from omari_us.utils import load_yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", required=True)
ap.add_argument("--regime", default="finetune", choices=["frozen", "finetune"])
ap.add_argument("--splits", default="artifacts/splits/split_manifest.csv")
ap.add_argument("--data-root", default=None, help="re-root image paths in the manifest")
ap.add_argument("--label-fraction", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--all-fractions", action="store_true")
ap.add_argument("--out", default="runs")
ap.add_argument("--demo", action="store_true")
ap.add_argument("--checkpoint", default=None, help="USF-MAE pretraining checkpoint (overrides config)")
a = ap.parse_args()

c = load_yaml(a.config)
df = load_split_manifest(a.splits, a.data_root)
jobs = [(f, s) for f in c.get("label_fractions", [1.0]) for s in c.get("seeds", [0])] if a.all_fractions else [(a.label_fraction, a.seed)]
for frac, seed in jobs:
    r = c["regimes"][a.regime]
    cfg = TrainConfig(model=c["model"], regime=a.regime, label_fraction=frac, seed=seed, out_dir=a.out,
                      loss=c.get("loss", "weighted_ce"), weight_scheme=c.get("weight_scheme", "inverse"), label_smoothing=c.get("label_smoothing", 0.1),
                      patience=c.get("patience", 6), input_overrides=c.get("input", {}), augment=c.get("augment", {}),
                      usf_mae_checkpoint=a.checkpoint or c.get("checkpoint"), usf_mae_global_pool=c.get("global_pool", "token"),
                      **{k: v for k, v in r.items()})
    if a.demo:
        cfg.epochs, cfg.max_train_batches, cfg.max_eval_batches, cfg.run_name = 2, 3, 2, cfg.name() + "_demo"
    s = train(cfg, df)
    print(s["run"], "val macro-F1", round(s["best_val_macro_f1"], 4), "test macro-F1", round(s["test"]["macro_f1"], 4))
