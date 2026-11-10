#!/usr/bin/env python
"""Write evaluation.json, training_curves.json and label_efficiency.json (no per-sample artifacts).
    python scripts/evaluate.py --config configs/export.yaml
"""
import argparse
import json
import os

import _path  # noqa: F401
from omari_us.datasets import load_split_manifest
from omari_us.export import Exporter
from omari_us.utils import load_yaml, write_json

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/export.yaml")
ap.add_argument("--data-root", default=None)
a = ap.parse_args()
cfg = load_yaml(a.config)
df = load_split_manifest(cfg["split_manifest"], a.data_root)
ex = Exporter(cfg)
ex.out.mkdir(parents=True, exist_ok=True)
written = {k: f() for k, f in (("evaluation", lambda: ex.export_evaluation(df)), ("training_curves", ex.export_curves), ("label_efficiency", ex.export_label_efficiency))}
mf = ex.out / "manifest.json"
if mf.exists():  # update an existing manifest in place
    m = json.load(open(mf))
    m.update({k: v for k, v in written.items() if v})
    write_json(m, mf)
print(written)
