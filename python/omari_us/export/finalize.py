"""Pure helpers that turn saved run artifacts into bundle schema 1.1 sections.

Used by `scripts/finalize_website_bundle.py` (correct an existing bundle without re-running inference) and by
`Exporter` (fresh exports). Nothing here runs a model; every number is read from summary.json / *.npz / CSV files.
"""
from __future__ import annotations

import glob
import json
import os
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ..utils import CLASSES, MODEL_IDS, NUM_CLASSES

LE_FRACTIONS: Tuple[float, ...] = (0.01, 0.05, 0.1, 0.25, 0.5, 1.0)
LE_SEEDS: Tuple[int, ...] = (0, 1, 2)


# ------------------------------------------------------------------------------------------------ run summaries
def is_demo_run(run_id: str, summary: Optional[Dict] = None) -> bool:
    return "_demo" in run_id or bool((summary or {}).get("demo_mode"))


def collect_run_summaries(pattern: str) -> List[Tuple[str, Dict]]:
    """[(run_id, summary)] for every summary.json matching `pattern` (a glob, or a runs/ directory), excluding demo runs."""
    if os.path.isdir(pattern):
        pattern = os.path.join(pattern, "*", "summary.json")
    out = []
    for f in sorted(glob.glob(pattern)):
        run_id = os.path.basename(os.path.dirname(f))
        with open(f, encoding="utf-8") as fh:
            s = json.load(fh)
        if is_demo_run(run_id, s):
            continue
        out.append((run_id, s))
    return out


def _pct(f: float) -> str:
    return f"{f * 100:g}%"


def _availability_text(present: Dict[float, List[int]], fractions: Sequence[float], seeds: Sequence[int]) -> str:
    cells = [(f, s) for f in fractions for s in present.get(f, [])]
    if not cells:
        return "No runs in this regime."
    if len(cells) == len(fractions) * len(seeds):
        return f"All {len(fractions)} label fractions x {len(seeds)} seeds were run."
    if len(cells) == 1:
        f, s = cells[0]
        return f"Only {_pct(f)} labels, seed {s}, was run; no low-label sweep."
    groups: List[Tuple[Tuple[int, ...], List[float]]] = []
    for f in fractions:
        key = tuple(present.get(f, []))
        if groups and groups[-1][0] == key:
            groups[-1][1].append(f)
        else:
            groups.append((key, [f]))
    parts = []
    for key, fs in groups:
        label = ", ".join(_pct(f) for f in fs)
        missing = [s for s in seeds if s not in key]
        if not key:
            parts.append(f"{label}: not run")
        elif not missing:
            parts.append(f"{label}: seeds {', '.join(map(str, key))}")
        else:
            parts.append(f"{label}: seed{'s' if len(key) > 1 else ''} {', '.join(map(str, key))} only "
                         f"(seed{'s' if len(missing) > 1 else ''} {', '.join(map(str, missing))} not run)")
    return "; ".join(parts) + "."


def build_label_efficiency(summaries: Iterable[Tuple[str, Dict]], regime: str = "finetune", models: Sequence[str] = MODEL_IDS,
                           fractions: Optional[Sequence[float]] = None, seeds: Optional[Sequence[int]] = None,
                           provenance: Optional[Dict] = None, notes: Optional[List[str]] = None) -> Dict:
    """Label-efficiency grid (schema 1.1): per model, fractions x seeds arrays of test macro-F1 / accuracy / run id.

    Missing runs are None (JSON null), never NaN. Every requested model is present even if it has no runs."""
    rows = []
    for run_id, s in summaries:
        if is_demo_run(run_id, s):
            continue
        c = s.get("config", {})
        if c.get("regime") != regime or c.get("model") not in models:
            continue
        rows.append({"model": c["model"], "fraction": float(c["label_fraction"]), "seed": int(c["seed"]), "run": run_id,
                     "macro_f1": float(s["test"]["macro_f1"]), "accuracy": float(s["test"]["accuracy"]),
                     "n_train_images": s.get("n_train_images"), "n_train_patients": s.get("n_train_patients")})
    fr = sorted(set(fractions or LE_FRACTIONS) | {r["fraction"] for r in rows})
    sd = sorted(set(seeds or LE_SEEDS) | {r["seed"] for r in rows})
    per, avail = {}, {}
    for mid in models:
        cell = {(r["fraction"], r["seed"]): r for r in rows if r["model"] == mid}
        grid = lambda key: [[(cell[(f, s)][key] if (f, s) in cell else None) for s in sd] for f in fr]  # noqa: E731
        per[mid] = {"macro_f1": grid("macro_f1"), "accuracy": grid("accuracy"), "runs": grid("run"),
                    "n_train_images": grid("n_train_images"), "n_train_patients": grid("n_train_patients")}
        avail[mid] = _availability_text({f: [s for s in sd if (f, s) in cell] for f in fr}, fr, sd)
    return {"provenance": provenance or {}, "regime": regime, "fractions": fr, "seeds": sd, "per_model": per,
            "availability": avail, "notes": list(notes or [])}


def seed_summary(summaries: Iterable[Tuple[str, Dict]], model: str, regime: str = "finetune", fraction: float = 1.0) -> Optional[Dict]:
    """Across-seed summary of test accuracy / macro-F1 for one model at one label fraction (sd = sample sd, ddof=1)."""
    rows = sorted(((int(s["config"]["seed"]), run_id, float(s["test"]["accuracy"]), float(s["test"]["macro_f1"]))
                   for run_id, s in summaries
                   if s.get("config", {}).get("model") == model and s["config"].get("regime") == regime
                   and abs(float(s["config"].get("label_fraction", -1)) - fraction) < 1e-9 and not is_demo_run(run_id, s)))
    if not rows:
        return None
    acc = np.array([r[2] for r in rows])
    f1 = np.array([r[3] for r in rows])
    sd = (lambda a: float(a.std(ddof=1)) if len(a) > 1 else None)
    return {"seeds": [r[0] for r in rows], "runs": [r[1] for r in rows], "accuracy": acc.tolist(), "macro_f1": f1.tolist(),
            "mean": {"accuracy": float(acc.mean()), "macro_f1": float(f1.mean())}, "sd": {"accuracy": sd(acc), "macro_f1": sd(f1)},
            "sd_definition": "sample standard deviation (ddof=1)", "regime": regime, "label_fraction": fraction}


# ------------------------------------------------------------------------------------------------ dataset stats
def _counts(d: pd.DataFrame) -> List[int]:
    return [int((d["label"] == c).sum()) for c in range(NUM_CLASSES)]


def enrich_dataset_stats(base: Dict, split_df: pd.DataFrame, near_dups: Optional[pd.DataFrame] = None,
                         root_near_dups: Optional[Dict] = None, script: str = "omari_us.export.finalize") -> Dict:
    """Add schema-1.1 fields to a 1.0 DatasetStats dict.

    split_df: split manifest (columns split ∈ train/val/test, label, Patient_num, US_Machine, Train).
    near_dups: splits/near_duplicates.csv (a, b = row indices into split_df; split_a/split_b).
    root_near_dups: the `near_duplicates` block notebook 01 wrote to ART/dataset_stats.json (method, threshold, groups...)."""
    out = json.loads(json.dumps(base))  # deep copy
    tr, va, te = (split_df[split_df["split"] == k] for k in ("train", "val", "test"))
    trv = split_df[split_df["split"].isin(["train", "val"])]
    out["class_counts"] = {"train": _counts(tr), "val": _counts(va), "test": _counts(te), "train_plus_val": _counts(trv)}
    pt = lambda d: set(d["Patient_num"])  # noqa: E731
    overlap = len(pt(tr) & pt(va)) + len(pt(tr) & pt(te)) + len(pt(va) & pt(te))
    out["split"] = {"train_images": int(len(tr)), "val_images": int(len(va)), "test_images": int(len(te)),
                    "train_patients": len(pt(tr)), "val_patients": len(pt(va)), "test_patients": len(pt(te)),
                    "patient_overlap": int(overlap),
                    "protocol": ("Authors' official patient-level Train/Test split kept unchanged; validation = patient-disjoint ~10% of the "
                                 "official training patients (StratifiedGroupKFold, seed 42). 'train' excludes 'val'; 'train_plus_val' is the "
                                 "authors' official training set.")}
    if "US_Machine" in split_df:
        out["machines_by_split"] = {k: {str(m): int(n) for m, n in d["US_Machine"].value_counts().sort_index().items()}
                                    for k, d in (("train", tr), ("val", va), ("test", te))}
        for k in out["machines_by_split"].values():
            for m in sorted(split_df["US_Machine"].unique()):
                k.setdefault(str(m), 0)
    if near_dups is not None:
        nd = near_dups
        labels = split_df["label"].to_numpy()
        # sanity: the CSV's split columns must agree with the manifest rows it indexes
        sp = split_df["split"].to_numpy()
        if not ((sp[nd["a"].to_numpy()] == nd["split_a"].to_numpy()).all() and (sp[nd["b"].to_numpy()] == nd["split_b"].to_numpy()).all()):
            raise ValueError("near_duplicates.csv indices do not match split_manifest.csv rows")
        pair = nd.apply(lambda r: "_".join(sorted([r["split_a"], r["split_b"]])), axis=1)
        cross = nd[nd["split_a"] != nd["split_b"]]
        cross_pair = pair[nd["split_a"] != nd["split_b"]]
        tvt = cross[cross_pair.isin(["test_train", "test_val"])]
        test_idx = set(tvt.loc[tvt["split_a"] == "test", "a"]).union(tvt.loc[tvt["split_b"] == "test", "b"])
        disc = float((labels[cross["a"].to_numpy()] != labels[cross["b"].to_numpy()]).mean()) if len(cross) else None
        involved = set(nd["a"]).union(nd["b"])
        r = dict(root_near_dups or {})
        out["near_duplicates"] = {
            "method": r.get("method", "dHash 8x8"), "threshold": r.get("threshold", int(nd["hamming"].max()) if len(nd) else None),
            "pairs": int(len(nd)), "groups": r.get("groups"), "images_in_groups": r.get("images_in_groups", len(involved)),
            "cross_split_pairs": int(len(tvt)),
            "cross_split_by_pair": {"train_val": int((cross_pair == "train_val").sum()), "train_test": int((cross_pair == "test_train").sum()),
                                    "val_test": int((cross_pair == "test_val").sum())},
            "test_images_involved": int(len(test_idx)),
            "label_discordant_fraction_cross_split": disc,
            "note": ("dHash (8x8, Hamming <= threshold) is a loose detector: it flags perceptually generic ultrasound frames, not only copies. "
                     "cross_split_pairs counts (train or val) <-> test pairs, as in notebook 01; cross_split_by_pair also lists train <-> val. "
                     "test_images_involved = distinct test images in a (train or val) <-> test pair. label_discordant_fraction_cross_split = "
                     "share of ALL cross-split pairs (incl. train <-> val) whose two images have different labels. No cross-split pair "
                     "shares a patient."),
        }
        if r.get("cross_split_pairs") is not None and int(r["cross_split_pairs"]) != int(len(tvt)):
            out["near_duplicates"]["note"] += f" (notebook 01 reported cross_split_pairs={r['cross_split_pairs']})"
    prov = dict(out.get("provenance") or {})
    prov["note"] = (prov.get("note", "") + "; schema-1.1 fields (val split, machines_by_split, near_duplicates) added by "
                    f"{script} from splits/split_manifest.csv, splits/near_duplicates.csv and notebook-01 dataset_stats.json").lstrip("; ")
    out["provenance"] = prov
    return out


# ------------------------------------------------------------------------------------------------ faithfulness
def parse_faithfulness_csv(path: str) -> List[Dict]:
    """Rows of notebook 06's faithfulness.csv (pandas 3-row MultiIndex header: metric / statistic / index names)."""
    df = pd.read_csv(path, header=[0, 1], index_col=[0, 1])
    rows = []
    for (model, method), r in df.iterrows():
        g = lambda m, s: float(r[(m, s)]) if pd.notna(r[(m, s)]) else None  # noqa: E731
        rows.append({"model": str(model), "method": str(method), "deletion_auc_mean": g("deletion_auc", "mean"),
                     "deletion_auc_sd": g("deletion_auc", "std"), "insertion_auc_mean": g("insertion_auc", "mean"),
                     "insertion_auc_sd": g("insertion_auc", "std")})
    return rows


# ------------------------------------------------------------------------------------------------ per-sample predictions
def load_predictions(npz_path: str) -> Dict[str, Dict]:
    """{image_id: {"probs": [...], "pred": int, "label": int}} from a run's *_predictions.npz (float32 values kept exactly)."""
    z = np.load(npz_path, allow_pickle=True)
    probs = z["probs"]
    out = {}
    for i, sid in enumerate(z["ids"]):
        p = probs[i]
        out[str(sid)] = {"probs": [float(v) for v in p], "pred": int(np.argmax(p)), "label": int(z["labels"][i])}
    return out


__all__ = ["LE_FRACTIONS", "LE_SEEDS", "build_label_efficiency", "collect_run_summaries", "enrich_dataset_stats", "is_demo_run",
           "load_predictions", "parse_faithfulness_csv", "seed_summary", "CLASSES"]
