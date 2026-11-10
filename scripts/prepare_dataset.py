#!/usr/bin/env python
"""Load FETAL_PLANES_DB metadata, build patient-level splits, detect near-duplicates, write manifests and statistics.

    python scripts/prepare_dataset.py --data-root /path/to/FETAL_PLANES_ZENODO --out artifacts/splits [--no-duplicates]
Writes: split_manifest.csv, dataset_stats.json (also copied to website/public/experiments/), near_duplicates.csv
"""
import argparse
import os
import shutil

import _path  # noqa: F401
from omari_us.datasets import check_patient_disjoint, dataset_stats, find_near_duplicates, make_splits, read_metadata, save_split_manifest
from omari_us.utils import write_json

ap = argparse.ArgumentParser()
ap.add_argument("--data-root", required=True)
ap.add_argument("--out", default="artifacts/splits")
ap.add_argument("--val-fraction", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
ap.add_argument("--max-hamming", type=int, default=4)
ap.add_argument("--no-duplicates", action="store_true")
ap.add_argument("--website", default="website/public/experiments")
a = ap.parse_args()

df = make_splits(read_metadata(a.data_root), a.val_fraction, a.seed)
print("patients per split:", check_patient_disjoint(df))
print(df.groupby(["split", "Plane"]).size().unstack(0))
dups = None
if not a.no_duplicates:
    dups = find_near_duplicates(df, a.max_hamming)
    os.makedirs(a.out, exist_ok=True)
    dups.to_csv(os.path.join(a.out, "near_duplicates.csv"), index=False)
    print(f"near-duplicate pairs (dHash ≤ {a.max_hamming} bits): {len(dups)}")
save_split_manifest(df, os.path.join(a.out, "split_manifest.csv"))
stats = dataset_stats(df, dups, a.max_hamming, script="scripts/prepare_dataset.py")
write_json(stats, os.path.join(a.out, "dataset_stats.json"), indent=1)
if a.website:
    os.makedirs(a.website, exist_ok=True)
    shutil.copy(os.path.join(a.out, "dataset_stats.json"), os.path.join(a.website, "dataset_stats.json"))
print("done:", a.out)
