"""Export schema tests: strict JSON writing, label-efficiency grid building, dataset-stat enrichment, and (when a
finalized bundle is installed) the schema-1.1 keys of website/public/experiments/manifest.json."""
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from omari_us.export import finalize as FZ
from omari_us.utils import json_sanitize, write_json

ROOT = Path(__file__).resolve().parents[1]
BUNDLE = ROOT / "website" / "public" / "experiments"


def _strict(path):
    def bad(c):
        raise ValueError(c)
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=bad)


def test_write_json_converts_nonfinite_to_null(tmp_path):
    obj = {"a": float("nan"), "b": [1.0, float("inf"), -float("inf")], "c": {"d": np.float32("nan"), "e": np.array([np.nan, 2.0])},
           "f": np.int64(3), "g": (np.float64(0.5),), "h": True}
    write_json(obj, tmp_path / "x.json")
    text = (tmp_path / "x.json").read_text(encoding="utf-8")
    assert "NaN" not in text and "Infinity" not in text
    d = _strict(tmp_path / "x.json")
    assert d == {"a": None, "b": [1.0, None, None], "c": {"d": None, "e": [None, 2.0]}, "f": 3, "g": [0.5], "h": True}


def test_json_sanitize_preserves_finite_values():
    assert json_sanitize([0.1, math.pi, {"k": 1}]) == [0.1, math.pi, {"k": 1}]


def _summary(model, frac, seed, f1, acc, regime="finetune", demo=False):
    return {"config": {"model": model, "regime": regime, "label_fraction": frac, "seed": seed}, "test": {"macro_f1": f1, "accuracy": acc},
            "n_train_images": int(1000 * frac), "n_train_patients": int(100 * frac), "demo_mode": demo}


def _write_runs(tmp_path):
    runs = {
        "densenet161_finetune_lf0.05_s0": _summary("densenet161", 0.05, 0, 0.81, 0.83),
        "densenet161_finetune_lf0.05_s1": _summary("densenet161", 0.05, 1, 0.78, 0.79),
        "densenet161_finetune_lf1_s0": _summary("densenet161", 1.0, 0, 0.94, 0.95),
        "vit_b16_finetune_lf1_s0": _summary("vit_b16", 1.0, 0, 0.941, 0.9465),
        "vit_b16_frozen_lf1_s0": _summary("vit_b16", 1.0, 0, 0.80, 0.82, regime="frozen"),
        "usf_mae_finetune_lf1_s0_demo": _summary("usf_mae", 1.0, 0, 0.11, 0.12),  # excluded by name
        "usf_mae_finetune_lf1_s1": _summary("usf_mae", 1.0, 1, 0.12, 0.13, demo=True),  # excluded by demo_mode
        "usf_mae_finetune_lf1_s2": _summary("usf_mae", 1.0, 2, 0.944, 0.9495),
    }
    for name, s in runs.items():
        (tmp_path / name).mkdir()
        (tmp_path / name / "summary.json").write_text(json.dumps(s), encoding="utf-8")
    return runs


def test_label_efficiency_grid_from_summaries(tmp_path):
    _write_runs(tmp_path)
    summ = FZ.collect_run_summaries(str(tmp_path / "*_lf*_s*" / "summary.json"))
    assert not any("_demo" in r or s.get("demo_mode") for r, s in summ)
    le = FZ.build_label_efficiency(summ, regime="finetune")
    assert le["regime"] == "finetune" and le["fractions"] == [0.01, 0.05, 0.1, 0.25, 0.5, 1.0] and le["seeds"] == [0, 1, 2]
    assert set(le["per_model"]) == {"densenet161", "vit_b16", "usf_mae"}
    d = le["per_model"]["densenet161"]
    i5, i100 = le["fractions"].index(0.05), le["fractions"].index(1.0)
    assert d["macro_f1"][i5] == [0.81, 0.78, None] and d["accuracy"][i5] == [0.83, 0.79, None]
    assert d["runs"][i100] == ["densenet161_finetune_lf1_s0", None, None]
    v = le["per_model"]["vit_b16"]
    assert v["macro_f1"][i100] == [0.941, None, None]  # frozen run not mixed in
    assert sum(x is not None for row in v["macro_f1"] for x in row) == 1
    assert le["availability"]["vit_b16"] == "Only 100% labels, seed 0, was run; no low-label sweep."
    u = le["per_model"]["usf_mae"]
    assert u["macro_f1"][i100] == [None, None, 0.944]  # demo runs dropped
    write_json(le, tmp_path / "le.json")
    assert _strict(tmp_path / "le.json")["per_model"]["vit_b16"]["accuracy"][0] == [None, None, None]


def test_seed_summary_uses_sample_sd(tmp_path):
    summ = [(f"usf_mae_finetune_lf1_s{i}", _summary("usf_mae", 1.0, i, f, f + 0.005)) for i, f in enumerate([0.94, 0.95, 0.96])]
    ss = FZ.seed_summary(summ, "usf_mae")
    assert ss["seeds"] == [0, 1, 2] and math.isclose(ss["mean"]["macro_f1"], 0.95)
    assert math.isclose(ss["sd"]["macro_f1"], 0.01)  # ddof=1


def test_enrich_dataset_stats_val_split_and_near_duplicates():
    df = pd.DataFrame({"Image_name": [f"im{i}" for i in range(6)], "Patient_num": [1, 1, 2, 3, 4, 4], "label": [0, 1, 0, 2, 0, 5],
                       "split": ["train", "train", "val", "test", "test", "test"], "US_Machine": ["A", "B", "A", "S10", "S10", "A"]})
    nd = pd.DataFrame({"a": [0, 0, 2, 1], "b": [1, 3, 4, 5], "hamming": [0, 2, 3, 4], "same_patient": [True, False, False, False],
                       "split_a": ["train", "train", "val", "train"], "split_b": ["train", "test", "test", "test"]})
    out = FZ.enrich_dataset_stats({"provenance": {}, "class_counts": {"train": [9], "test": [9]}}, df, nd, {"method": "dHash 8x8", "threshold": 4, "groups": 1})
    assert out["class_counts"]["train"] == [1, 1, 0, 0, 0, 0] and out["class_counts"]["val"] == [1, 0, 0, 0, 0, 0]
    assert out["class_counts"]["train_plus_val"] == [2, 1, 0, 0, 0, 0]
    assert out["split"]["val_images"] == 1 and out["split"]["patient_overlap"] == 0
    assert out["machines_by_split"]["val"] == {"A": 1, "B": 0, "S10": 0}
    n = out["near_duplicates"]
    assert n["pairs"] == 4 and n["cross_split_pairs"] == 3 and n["cross_split_by_pair"] == {"train_val": 0, "train_test": 2, "val_test": 1}
    assert n["test_images_involved"] == 3 and math.isclose(n["label_discordant_fraction_cross_split"], 2 / 3)


@pytest.mark.skipif(not (BUNDLE / "manifest.json").exists(), reason="no finalized bundle installed in website/public/experiments")
def test_finalized_manifest_schema_keys():
    m = _strict(BUNDLE / "manifest.json")
    if m["schema_version"] != "1.1":
        pytest.skip("installed bundle is not schema 1.1")
    for k in ("provenance_notes", "sample_predictions_source", "faithfulness", "label_efficiency", "dataset_stats", "evaluation"):
        assert m.get(k), k
    assert all(isinstance(x["notes"], str) and x["notes"] for x in m["models"])
    usf = next(x for x in m["models"] if x["id"] == "usf_mae")
    assert usf["pretraining"]["epochs"] == 500 and usf["pretraining"]["sha256"].startswith("c659a061")
    for s in m["samples"]:
        assert s["split"] == "test"
        assert all(a["method"] != "chefer" and a["method_label"] and a["layer_label"] for a in s.get("attributions", []))
        for r in s.get("reconstructions", []):
            assert r["provenance"]["checkpoint_id"] == "usf_mae_500ep.pth"
    le = _strict(BUNDLE / m["label_efficiency"])
    assert le["regime"] == "finetune" and set(le["per_model"]) == {"densenet161", "vit_b16", "usf_mae"}
    for k in ("faithfulness", "dataset_stats", "evaluation", "training_curves"):
        _strict(BUNDLE / m[k])  # strict JSON
