#!/usr/bin/env python
"""Validate an artifact bundle: schema version, referenced files exist, every model output carries provenance,
checkpoint hashes are consistent, no sample label/prediction is out of range, and no JSON file contains NaN/Infinity.

For schema 1.1 bundles it also checks the finalization contract (scripts/finalize_website_bundle.py): 1.1 keys, all
sample ids in the test split, sample predictions == the headline runs' test_predictions.npz, label-efficiency values ==
summary.json, and that the USF-MAE checkpoint is usf_mae_500ep.pth everywhere except samples[].reconstructions_legacy.
Those artifact cross-checks need --artifacts (default content/omari_artifacts/omari_artifacts, skipped if absent).

    python scripts/check_provenance.py [website/public/experiments] [--artifacts content/omari_artifacts/omari_artifacts]
Exit code 0 = OK (or no bundle present), 1 = problems found.
"""
import argparse
import glob
import json
import os
import re
import sys

ap = argparse.ArgumentParser()
ap.add_argument("root", nargs="?", default="website/public/experiments")
ap.add_argument("--artifacts", default="content/omari_artifacts/omari_artifacts")
a = ap.parse_args()
root = a.root
USF_100 = ("usf_mae_100ep", "f815c629878c17136985af9f4fdc81c2cfa02a94e4d992c026699957f75ccb66")
USF_500_SHA = "c659a06170b0f6967ae7f6d066681b8dea2c122a5f325bb8a82df11ffb3a2533"

mf = os.path.join(root, "manifest.json")
if not os.path.exists(mf):
    print(f"no bundle at {root} (website will show empty states) — OK")
    sys.exit(0)
errs = []


def strict_load(path):
    def bad(c):
        raise ValueError(f"non-finite constant {c}")
    with open(path, encoding="utf-8") as f:
        return json.load(f, parse_constant=bad)


# -- every JSON file in the bundle must be strict JSON (browsers reject NaN/Infinity)
for f in glob.glob(os.path.join(root, "**", "*.json"), recursive=True):
    try:
        strict_load(f)
    except ValueError as e:
        errs.append(f"{os.path.relpath(f, root)}: {e}")

m = strict_load(mf)
v11 = str(m.get("schema_version")) == "1.1"
final = v11 and "finalizer" in m  # bundle produced by scripts/finalize_website_bundle.py: full contract
if not str(m.get("schema_version", "")).startswith("1."):
    errs.append(f"schema_version {m.get('schema_version')}")
shas = {x["id"]: x.get("checkpoint_sha256") for x in m.get("models", [])}
K = len(m.get("classes", []))


def need(path, what):
    if not os.path.exists(os.path.join(root, path)):
        errs.append(f"missing file for {what}: {path}")


# -- every relative file path referenced anywhere in the manifest exists
PATH_RE = re.compile(r"^[\w.\-/]+\.(png|json|bin|txt|csv)$")


def walk_paths(o, where="manifest"):
    if isinstance(o, dict):
        for k, v in o.items():
            if k == "provenance":
                continue  # provenance params may name source files outside the bundle
            walk_paths(v, f"{where}.{k}")
    elif isinstance(o, list):
        for i, v in enumerate(o):
            walk_paths(v, f"{where}[{i}]")
    elif isinstance(o, str) and PATH_RE.match(o) and not o.startswith("/"):
        need(o, where)


walk_paths(m)

for s in m.get("samples", []):
    if not (0 <= s["label"] < K):
        errs.append(f"{s['id']}: label out of range")
    for key in ("predictions", "predictions_display_png"):
        for mid, p in (s.get(key) or {}).items():
            if len(p["probs"]) != K or abs(sum(p["probs"]) - 1) > 1e-3 or p["pred"] != max(range(K), key=lambda i: p["probs"][i]):
                errs.append(f"{s['id']}/{key}/{mid}: probabilities invalid")
            if mid not in shas:
                errs.append(f"{s['id']}: prediction from model {mid} not declared in manifest.models")
    for at in s.get("attributions", []):
        pv = at.get("provenance", {})
        if not pv.get("script") or not pv.get("checkpoint_sha256"):
            errs.append(f"{s['id']}: attribution without provenance")
        elif shas.get(at["model"]) and pv["checkpoint_sha256"] != shas[at["model"]]:
            errs.append(f"{s['id']}: attribution checkpoint hash differs from manifest")
        if v11:
            if at["method"] == "chefer":
                errs.append(f"{s['id']}: method id 'chefer' in a 1.1 bundle (should be grad_rollout)")
            if not at.get("method_label") or not at.get("layer_label"):
                errs.append(f"{s['id']}: attribution without method_label/layer_label")
    for key in ("reconstructions", "reconstructions_legacy"):
        for r in s.get(key, []) or []:
            if not r.get("provenance", {}).get("checkpoint_sha256"):
                errs.append(f"{s['id']}: {key} entry without checkpoint provenance")
    for p in s.get("perturbations", []):
        if not p.get("provenance"):
            errs.append(f"{s['id']}: perturbation without provenance")
        elif v11 and not p["provenance"].get("params", {}).get("checkpoints"):
            errs.append(f"{s['id']}: perturbation provenance without checkpoint ids")

files = {}
for key in ("evaluation", "training_curves", "label_efficiency", "dataset_stats", "faithfulness"):
    if m.get(key) and os.path.exists(os.path.join(root, m[key])):
        files[key] = strict_load(os.path.join(root, m[key]))
        if not files[key].get("provenance"):
            errs.append(f"{key}: no provenance")

# ---------------------------------------------------------------- schema 1.1 finalized-bundle contract
if final:
    for k in ("provenance_notes", "sample_predictions_source", "faithfulness", "label_efficiency", "dataset_stats", "evaluation"):
        if not m.get(k):
            errs.append(f"manifest: missing 1.1 key {k}")
    for x in m.get("models", []):
        if not isinstance(x.get("notes"), str) or not x["notes"]:
            errs.append(f"models.{x['id']}: notes must be a non-empty string")
    usf = next((x for x in m.get("models", []) if x["id"] == "usf_mae"), None)
    if usf:
        pre = usf.get("pretraining") or {}
        if pre.get("sha256") != USF_500_SHA or pre.get("epochs") != 500:
            errs.append("models.usf_mae.pretraining is not the 500-epoch checkpoint")
        if "500ep" not in usf.get("initialization", ""):
            errs.append("models.usf_mae.initialization does not name usf_mae_500ep.pth")
        if "OpenUS-46" not in usf.get("notes", "") or "FETAL_PLANES_DB" not in usf.get("notes", ""):
            errs.append("models.usf_mae.notes lacks the pretraining-overlap disclosure")
    le = files.get("label_efficiency", {})
    if le.get("regime") != "finetune":
        errs.append(f"label_efficiency.regime = {le.get('regime')}")
    for mid in ("densenet161", "vit_b16", "usf_mae"):
        pm = (le.get("per_model") or {}).get(mid)
        if not pm:
            errs.append(f"label_efficiency: model {mid} missing")
            continue
        for k in ("macro_f1", "accuracy", "runs"):
            g = pm.get(k)
            if not g or len(g) != len(le["fractions"]) or any(len(r) != len(le["seeds"]) for r in g):
                errs.append(f"label_efficiency.{mid}.{k}: not a fractions x seeds grid")
        if not (le.get("availability") or {}).get(mid):
            errs.append(f"label_efficiency.availability.{mid} missing")
    ds = files.get("dataset_stats", {})
    for k in ("val", "train_plus_val"):
        if k not in ds.get("class_counts", {}):
            errs.append(f"dataset_stats.class_counts.{k} missing")
    for k in ("val_images", "val_patients", "protocol"):
        if k not in ds.get("split", {}):
            errs.append(f"dataset_stats.split.{k} missing")
    for k in ("pairs", "cross_split_by_pair", "test_images_involved"):
        if k not in (ds.get("near_duplicates") or {}):
            errs.append(f"dataset_stats.near_duplicates.{k} missing")
    ev = files.get("evaluation", {})
    if not (ev.get("seed_summary") or {}).get("usf_mae"):
        errs.append("evaluation.seed_summary.usf_mae missing")
    for mid, pm in (ev.get("per_model") or {}).items():
        for k in ("temperature", "ece_after_temperature"):
            if k not in pm:
                errs.append(f"evaluation.per_model.{mid}.{k} missing")
    fa = files.get("faithfulness", {})
    if not fa.get("rows") or any(r["method"] == "chefer" for r in fa["rows"]):
        errs.append("faithfulness.json: rows missing or still named chefer")

    # USF-MAE 100ep may appear only inside samples[].reconstructions_legacy
    def scan(o, where):
        if isinstance(o, dict):
            for k, v in o.items():
                if k == "reconstructions_legacy":
                    continue
                scan(v, f"{where}.{k}")
        elif isinstance(o, list):
            for i, v in enumerate(o):
                scan(v, f"{where}[{i}]")
        elif isinstance(o, str) and any(t in o for t in USF_100):
            if not (where.startswith("manifest.provenance_notes") or where.startswith("manifest.finalizer")):
                errs.append(f"{where}: references the 100-epoch USF-MAE checkpoint outside reconstructions_legacy")

    scan({k: v for k, v in m.items()}, "manifest")
    for key, d in files.items():
        scan(d, key)
    for s in m.get("samples", []):
        for r in s.get("reconstructions", []):
            pv = r.get("provenance", {})
            if pv.get("checkpoint_id") != "usf_mae_500ep.pth" or pv.get("checkpoint_sha256") != USF_500_SHA:
                errs.append(f"{s['id']}: primary reconstruction not from usf_mae_500ep.pth")

    # ---------------------------------------------------------------- cross-checks against the saved artifacts
    art = a.artifacts
    if art and os.path.isdir(os.path.join(art, "runs")):
        import numpy as np
        import pandas as pd

        split = pd.read_csv(os.path.join(art, "splits", "split_manifest.csv"), usecols=["Image_name", "split"])
        split_of = dict(zip(split["Image_name"], split["split"]))
        npz = {}
        for x in m["models"]:
            z = np.load(os.path.join(art, "runs", x["checkpoint_id"].split("/")[0], "test_predictions.npz"), allow_pickle=True)
            npz[x["id"]] = {str(i): z["probs"][k] for k, i in enumerate(z["ids"])}
        for s in m["samples"]:
            if split_of.get(s["id"]) != "test":
                errs.append(f"{s['id']}: not in the test split")
            for mid, p in s["predictions"].items():
                ref = npz[mid].get(s["id"])
                if ref is None or np.abs(np.asarray(p["probs"]) - ref.astype(np.float64)).max() > 1e-7 or p["pred"] != int(ref.argmax()):
                    errs.append(f"{s['id']}/{mid}: prediction differs from test_predictions.npz")
        summ = {}
        for f in glob.glob(os.path.join(art, "runs", "*", "summary.json")):
            summ[os.path.basename(os.path.dirname(f))] = json.load(open(f, encoding="utf-8"))
        n_cells = 0
        for mid, pm in le.get("per_model", {}).items():
            for i, fr in enumerate(le["fractions"]):
                for j, sd in enumerate(le["seeds"]):
                    run = pm["runs"][i][j]
                    if run is None:
                        exists = [r for r, s in summ.items() if "_demo" not in r and s["config"]["model"] == mid and s["config"]["regime"] == "finetune"
                                  and abs(s["config"]["label_fraction"] - fr) < 1e-9 and s["config"]["seed"] == sd]
                        if exists or pm["macro_f1"][i][j] is not None or pm["accuracy"][i][j] is not None:
                            errs.append(f"label_efficiency {mid} {fr} s{sd}: null but run exists ({exists})")
                        continue
                    s = summ.get(run)
                    n_cells += 1
                    if (s is None or s["config"]["model"] != mid or abs(s["config"]["label_fraction"] - fr) > 1e-9 or s["config"]["seed"] != sd
                            or pm["macro_f1"][i][j] != s["test"]["macro_f1"] or pm["accuracy"][i][j] != s["test"]["accuracy"]):
                        errs.append(f"label_efficiency {mid} {fr} s{sd}: value differs from {run}/summary.json")
        print(f"artifact cross-checks: {len(m['samples'])} samples vs npz, {n_cells} label-efficiency cells vs summary.json")
    else:
        print(f"artifact cross-checks skipped (no runs/ under {art})")

print(f"bundle {m['bundle_id']} (schema {m.get('schema_version')}): {len(m.get('samples', []))} samples, models {list(shas)}")
if errs:
    print("\n".join("ERROR " + e for e in errs[:80]))
    print(f"{len(errs)} problem(s)")
    sys.exit(1)
print("provenance check passed")
