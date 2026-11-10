#!/usr/bin/env python
"""Write the website artifact bundle from trained checkpoints. See configs/export.yaml and docs/ARTIFACT_SCHEMA.md.
    python scripts/export_website_artifacts.py --config configs/export.yaml [--clean] [--data-root ...]
"""
import argparse
import sys

import _path  # noqa: F401
from omari_us.datasets import load_split_manifest
from omari_us.export import Exporter
from omari_us.utils import load_yaml

ap = argparse.ArgumentParser()
ap.add_argument("--config", default="configs/export.yaml")
ap.add_argument("--data-root", default=None)
ap.add_argument("--clean", action="store_true", help="delete the output folder first")
a = ap.parse_args()
cfg = load_yaml(a.config)
cfg["_command"] = " ".join(sys.argv)
df = load_split_manifest(cfg["split_manifest"], a.data_root or (cfg.get("data_root") if "$" not in str(cfg.get("data_root")) else None))
Exporter(cfg).run(df, clean=a.clean)
