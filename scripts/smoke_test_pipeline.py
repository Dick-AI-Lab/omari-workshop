#!/usr/bin/env python
"""End-to-end pipeline check WITHOUT real data or pretrained weights.

Synthetic grey images → patient-level splits → 2-epoch demo training of all three models from RANDOM
initialisation (USF-MAE from a random MAE checkpoint) → full artifact export → provenance check.
Output goes to a temporary directory (default /tmp/omari_smoke) and must never be copied into
website/public/experiments: the numbers are meaningless by construction.

    python scripts/smoke_test_pipeline.py [--out /tmp/omari_smoke]
"""
import argparse
import os
import subprocess
import sys

import _path  # noqa: F401
import torch

from omari_us.datasets import make_splits, save_split_manifest, synthetic_metadata
from omari_us.export import Exporter
from omari_us.models.usf_mae import mae_vit_base_patch16_dec512d8b
from omari_us.training import TrainConfig, train

ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/tmp/omari_smoke")
ap.add_argument("--samples", type=int, default=2)
a = ap.parse_args()
assert "website/public" not in os.path.abspath(a.out), "refusing to write smoke outputs into the website"
torch.set_num_threads(max(1, os.cpu_count() or 1))
os.makedirs(a.out, exist_ok=True)
df = make_splits(synthetic_metadata(60, (2, 5), seed=0, image_dir=os.path.join(a.out, "Images"), size=(200, 150)), 0.15, 0)
save_split_manifest(df, os.path.join(a.out, "split_manifest.csv"))
ck = os.path.join(a.out, "random_mae.pth")
torch.save({"model": mae_vit_base_patch16_dec512d8b().state_dict()}, ck)
runs = {}
for mid in ("densenet161", "vit_b16", "usf_mae"):
    cfg = TrainConfig(model=mid, regime="finetune" if mid != "vit_b16" else "frozen", epochs=2, batch_size=4, max_train_batches=2, max_eval_batches=2,
                      num_workers=0, out_dir=os.path.join(a.out, "runs"), pretrained=False, usf_mae_checkpoint=ck if mid == "usf_mae" else None, amp=False)
    s = train(cfg, df, progress=False)
    runs[mid] = os.path.join(a.out, "runs", s["run"])
    print("trained", s["run"])
export_cfg = {
    "bundle_id": "SMOKE-TEST-random-weights", "output_dir": os.path.join(a.out, "bundle"), "max_side": 200,
    "models": {m: {"checkpoint": os.path.join(r, "best.pt")} for m, r in runs.items()},
    "samples": {"split": "test", "per_class": 1, "disagreements": 1, "misclassified_per_model": 0, "max_samples": a.samples, "seed": 0},
    "features": {"enabled": True, "channels": 4}, "attention": {"enabled": True, "full_for": "opening", "blocks_full": [11]},
    "attributions": {"enabled": True, "methods": "all", "all_layers": False}, "reconstructions": {"enabled": True, "ratios": [0.75], "seeds": [0]},
    "perturbations": {"enabled": True, "grid": [["contrast", 1.5, "contrast ×1.5"], ["occlude_annotations", 0.92, "overlays inpainted"]], "occlude_top": True},
    "evaluation": {"runs": runs, "bootstrap": 50, "label_efficiency_glob": os.path.join(a.out, "runs", "*", "summary.json")},
}
export_cfg["models"]["usf_mae"]["pretrain_checkpoint"] = ck
Exporter(export_cfg).run(df, clean=True)
r = subprocess.run([sys.executable, os.path.join(os.path.dirname(__file__), "check_provenance.py"), export_cfg["output_dir"]])
sys.exit(r.returncode)
