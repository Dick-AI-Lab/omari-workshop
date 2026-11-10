#!/usr/bin/env python
"""Validate an artifact bundle: schema version, referenced files exist, every model output carries provenance,
checkpoint hashes are consistent, and no sample label/prediction is out of range.
    python scripts/check_provenance.py website/public/experiments
Exit code 0 = OK (or no bundle present), 1 = problems found.
"""
import json
import os
import sys

root = sys.argv[1] if len(sys.argv) > 1 else "website/public/experiments"
mf = os.path.join(root, "manifest.json")
if not os.path.exists(mf):
    print(f"no bundle at {root} (website will show empty states) — OK")
    sys.exit(0)
m = json.load(open(mf))
errs = []
if not str(m.get("schema_version", "")).startswith("1."):
    errs.append(f"schema_version {m.get('schema_version')}")
shas = {x["id"]: x.get("checkpoint_sha256") for x in m.get("models", [])}
K = len(m.get("classes", []))


def need(path, what):
    if not os.path.exists(os.path.join(root, path)):
        errs.append(f"missing file for {what}: {path}")


for s in m.get("samples", []):
    need(s["image"], s["id"])
    if not (0 <= s["label"] < K):
        errs.append(f"{s['id']}: label out of range")
    for mid, p in s.get("predictions", {}).items():
        if len(p["probs"]) != K or abs(sum(p["probs"]) - 1) > 1e-3:
            errs.append(f"{s['id']}/{mid}: probabilities invalid")
        if mid not in shas:
            errs.append(f"{s['id']}: prediction from model {mid} not declared in manifest.models")
    for a in s.get("attributions", []):
        need(a["png"], s["id"])
        pv = a.get("provenance", {})
        if not pv.get("script") or not pv.get("checkpoint_sha256"):
            errs.append(f"{s['id']}: attribution without provenance")
        elif shas.get(a["model"]) and pv["checkpoint_sha256"] != shas[a["model"]]:
            errs.append(f"{s['id']}: attribution checkpoint hash differs from manifest")
    for r in s.get("reconstructions", []):
        need(r["reconstruction_png"], s["id"])
        if not r.get("provenance", {}).get("checkpoint_sha256"):
            errs.append(f"{s['id']}: reconstruction without checkpoint provenance")
    for p in s.get("perturbations", []):
        need(p["image_png"], s["id"])
        if not p.get("provenance"):
            errs.append(f"{s['id']}: perturbation without provenance")
for key in ("evaluation", "training_curves", "label_efficiency", "dataset_stats"):
    if m.get(key):
        need(m[key], key)
        d = json.load(open(os.path.join(root, m[key])))
        if not d.get("provenance"):
            errs.append(f"{key}: no provenance")
print(f"bundle {m['bundle_id']}: {len(m.get('samples', []))} samples, models {list(shas)}")
if errs:
    print("\n".join("ERROR " + e for e in errs[:50]))
    sys.exit(1)
print("provenance check passed")
