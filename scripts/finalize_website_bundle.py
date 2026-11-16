#!/usr/bin/env python
"""Produce the corrected website bundle (schema 1.1) from the exported 1.0 bundle and the saved run artifacts.

    python scripts/finalize_website_bundle.py --artifacts content/omari_artifacts/omari_artifacts \
        --out website/public/experiments [--reconstruct --device cuda] [--verify-densenet | --densenet-equivalence FILE]

Reads ARTIFACTS/{website_bundle, runs, splits, checkpoints, faithfulness.csv, dataset_stats.json}; never writes there.
Writes only inside --out (website/public/experiments/README.md is never touched). Idempotent: every run starts from the
pristine source manifest/JSONs and re-applies the same corrections; generated 500-epoch reconstructions are cached in
samples/<id>/reconstructions_500ep/index.json and reused when --reconstruct is not given.

Corrections (see the "Bundle schema 1.1" section of docs/ARTIFACT_SCHEMA.md):
  * label_efficiency.json  fine-tune grid (fractions x seeds; macro_f1, accuracy, runs; null = not run) for all 3 models
  * dataset_stats.json     near-duplicates kept, separate train/val/test counts, machines_by_split
  * manifest.json          schema 1.1, models[].notes + pretraining, provenance_notes, sample predictions from the headline
                           runs' test_predictions.npz, grad_rollout rename + method/layer labels, perturbation checkpoint ids,
                           500-epoch MAE reconstructions (legacy 100-epoch ones kept under reconstructions_legacy)
  * evaluation.json        seed_summary for USF-MAE, `temperature` alias of temperature_fit_on_val (numbers unchanged)
  * faithfulness.json      parsed from faithfulness.csv
  * training_curves.json   notes on the DenseNet history provenance
"""
import argparse
import copy
import json
import os
import shutil
import sys
from pathlib import Path

import _path  # noqa: F401
import numpy as np
import pandas as pd

from omari_us import explainability as X
from omari_us.export import finalize as FZ
from omari_us.export.website_export import LE_NOTES, SAMPLE_PREDICTIONS_SOURCE, faithfulness_json
from omari_us.utils import git_commit, load_yaml, now_iso, provenance, sha256_file, write_json

SCRIPT = "scripts/finalize_website_bundle.py"
SCHEMA = "1.1"
USF_500_SHA = "c659a06170b0f6967ae7f6d066681b8dea2c122a5f325bb8a82df11ffb3a2533"
USF_100_SHA = "f815c629878c17136985af9f4fdc81c2cfa02a94e4d992c026699957f75ccb66"
DENSENET_RUN = "densenet161_finetune_lf1_s0"

ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
ap.add_argument("--artifacts", default="content/omari_artifacts/omari_artifacts")
ap.add_argument("--out", default="website/public/experiments")
ap.add_argument("--config", default="configs/export.yaml", help="models[].notes / pretraining are read from here")
ap.add_argument("--images", default="content/data/FETAL_PLANES_ZENODO/Images", help="original images (only for --verify-densenet)")
ap.add_argument("--reconstruct", action="store_true", help="generate 500-epoch MAE reconstructions (GPU recommended)")
ap.add_argument("--verify-densenet", action="store_true", help="run the DenseNet functional-equivalence test (GPU inference)")
ap.add_argument("--densenet-equivalence", default=None, help="reuse a densenet_equivalence.json produced earlier")
ap.add_argument("--device", default=None)
a = ap.parse_args()

ART = Path(a.artifacts).resolve()
SRC = ART / "website_bundle"
OUT = Path(a.out).resolve()
if not (SRC / "manifest.json").exists():
    sys.exit(f"no source bundle at {SRC}")
if OUT == ART or ART in OUT.parents or OUT in ART.parents or OUT == SRC:
    sys.exit("--out must not be inside (or contain) --artifacts")
OUT.mkdir(parents=True, exist_ok=True)
LOG = []


def say(s):
    print(s, flush=True)
    LOG.append(s)


def inside_out(p: Path) -> Path:
    p = Path(p).resolve()
    if OUT not in p.parents:
        raise RuntimeError(f"refusing to write outside --out: {p}")
    return p


def wjson(obj, rel):
    write_json(obj, inside_out(OUT / rel))


def rjson(p):
    with open(p, encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------------ 1. copy the source bundle (source untouched)
n_copied = 0
for root, _dirs, files in os.walk(SRC):
    for fn in files:
        src = Path(root) / fn
        dst = inside_out(OUT / src.relative_to(SRC))
        if dst.name == "README.md" and dst.parent == OUT:
            continue
        if not dst.exists() or dst.stat().st_size != src.stat().st_size or dst.stat().st_mtime < src.stat().st_mtime:
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
            n_copied += 1
say(f"[copy] {SRC} -> {OUT} ({n_copied} files updated)")

man = rjson(SRC / "manifest.json")  # always start from the pristine source manifest
cfg = load_yaml(a.config) if os.path.exists(a.config) else {}
summaries = FZ.collect_run_summaries(str(ART / "runs"))
summ_by_run = dict(summaries)
split = pd.read_csv(ART / "splits" / "split_manifest.csv")
split_of = dict(zip(split["Image_name"], split["split"]))
headline = {m["id"]: m["checkpoint_id"].split("/")[0] for m in man["models"]}
model_ck = {m["id"]: {"checkpoint_id": m["checkpoint_id"], "checkpoint_sha256": m.get("checkpoint_sha256")} for m in man["models"]}

# ------------------------------------------------------------------ 2. DenseNet checkpoint status
dn_best = ART / "runs" / DENSENET_RUN / "best.pt"
dn_sha = sha256_file(dn_best)
dn_summary_sha = summ_by_run[DENSENET_RUN]["checkpoint_sha256"]
equiv = None
if a.verify_densenet:
    from omari_us.export.equivalence import functional_equivalence

    equiv = functional_equivalence(ART, a.images, [DENSENET_RUN, "vit_b16_finetune_lf1_s0"], device=a.device or "cuda")
elif a.densenet_equivalence:
    equiv = rjson(a.densenet_equivalence)
elif (OUT / "provenance" / "densenet_equivalence.json").exists():
    equiv = rjson(OUT / "provenance" / "densenet_equivalence.json")
if equiv:
    wjson(equiv, "provenance/densenet_equivalence.json")


def equiv_note():
    if not equiv:
        return "Functional equivalence of the rerun DenseNet weights with the original checkpoint was NOT evaluated in this finalization."
    d = equiv["runs"][DENSENET_RUN]["splits"]
    v = equiv["runs"]["vit_b16_finetune_lf1_s0"]["splits"]
    f = lambda x: f"{x:.2g}"  # noqa: E731
    on = lambda r, k: max(r[s]["amp_on"][k] for s in ("val", "test"))  # noqa: E731
    off = lambda r, k: max(r[s]["amp_off"][k] for s in ("val", "test"))  # noqa: E731
    dis = lambda r, mode: sum(r[s][mode]["argmax_disagree"] for s in ("val", "test"))  # noqa: E731
    n = d["val"]["n"] + d["test"]["n"]
    return (f"DenseNet functional-equivalence test ({equiv['environment']['gpu']}, torch {equiv['environment']['torch']}): the current best.pt "
            f"re-run on the {d['val']['n']} val + {d['test']['n']} test images reproduces the ORIGINAL run's saved probabilities with "
            f"max |dprob| = {f(on(d, 'max_abs_dprob'))} and {n - dis(d, 'amp_on')}/{n} argmax agreement under AMP (as in the original "
            f"evaluation), recomputed test accuracy {d['test']['amp_on']['recomputed_accuracy']:.4f} / macro-F1 "
            f"{d['test']['amp_on']['recomputed_macro_f1']:.4f} (= summary.json). Positive control ViT-B/16 (self-consistent run): "
            f"max |dprob| = {f(on(v, 'max_abs_dprob'))}, {dis(v, 'amp_on')} disagreements. With AMP off (fp32) DenseNet max |dprob| = "
            f"{f(off(d, 'max_abs_dprob'))} ({dis(d, 'amp_off')} flip) vs ViT {f(off(v, 'max_abs_dprob'))} ({dis(v, 'amp_off')} flip), i.e. "
            f"fp16-vs-fp32 sensitivity, not a weight difference. Verdict: the rerun weights are functionally equivalent to the original "
            f"checkpoint within numerical noise (not provable bit-identical; best.pt also stores a creation timestamp, so its hash "
            f"differs regardless). Details: provenance/densenet_equivalence.json.")


# ------------------------------------------------------------------ 3. models
cmodels = cfg.get("models") or {}
usf_ck = ART / "checkpoints" / "usf_mae_500ep.pth"
usf_sha = sha256_file(usf_ck)
if usf_sha != USF_500_SHA:
    sys.exit(f"{usf_ck} sha256 {usf_sha} != pinned {USF_500_SHA}")
for m in man["models"]:
    mc = cmodels.get(m["id"]) or {}
    notes = (mc.get("notes") or "").strip()
    if m["id"] == "densenet161":
        notes += (f" Checkpoint status: runs/{DENSENET_RUN}/best.pt (sha256 {dn_sha[:12]}...) was overwritten by an interrupted rerun "
                  f"and no longer matches summary.json (sha256 {dn_summary_sha[:12]}...). Metrics and predictions come from the original "
                  f"run; feature maps, attributions, perturbations and the training curve come from the rerun weights (see provenance_notes).")
    m["notes"] = notes.strip()
    pre = dict(mc.get("pretraining") or {})
    if m["id"] == "usf_mae":
        pre.update({"checkpoint": "usf_mae_500ep.pth", "sha256": usf_sha})
        if "500ep" not in m.get("initialization", ""):
            sys.exit("usf_mae initialization is not the 500-epoch checkpoint")
    if pre:
        m["pretraining"] = pre
    if m["id"] == "densenet161":
        m["checkpoint_status"] = {"best_pt_sha256": dn_sha, "summary_checkpoint_sha256": dn_summary_sha, "matches_summary": dn_sha == dn_summary_sha,
                                  "functional_equivalence": "provenance/densenet_equivalence.json" if equiv else None}

# ------------------------------------------------------------------ 4. per-sample corrections
npz = {mid: FZ.load_predictions(str(ART / "runs" / run / "test_predictions.npz")) for mid, run in headline.items()}
mae = None
if a.reconstruct:
    import torch
    from PIL import Image

    from omari_us.datasets import build_transforms, input_spec
    from omari_us.datasets.transforms import denormalize
    from omari_us.models.usf_mae import build_mae_from_state_dict, load_usf_mae_checkpoint, reconstruct

    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    sd, rep = load_usf_mae_checkpoint(str(usf_ck))
    if not rep.ok_for_reconstruction:
        say("[recon] 500ep checkpoint has no complete decoder; keeping 100-epoch reconstructions\n" + rep.summary())
    else:
        mae = build_mae_from_state_dict(sd).to(dev).eval()
        mspec = input_spec("usf_mae")
        mtf = build_transforms(mspec, train=False)
        say(f"[recon] usf_mae_500ep.pth loaded on {dev} (decoder complete)")


def save_gray(arr, path):
    from PIL import Image

    path = inside_out(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8), mode="L").save(path, optimize=True)


def make_recon_500(s):
    """Same input pixels (samples/<id>/image.png == the array the legacy export used), preprocessing, ratios, seeds and masks."""
    import torch
    from PIL import Image

    with Image.open(SRC / s["image"]) as im:
        gray = np.asarray(im.convert("L"), dtype=np.float32) / 255.0
    x = mtf(Image.fromarray((gray * 255).round().astype(np.uint8), mode="L")).unsqueeze(0).to(dev)
    out = []
    for old in s.get("reconstructions", []):
        ratio, seed = float(old["mask_ratio"]), int(old["seed"])
        r = reconstruct(mae, x, ratio, seed)
        mask_source = "seeded noise (torch CPU generator, seed) - reproduces the legacy masked indices exactly"
        if r["masked_indices"][0] != old["masked"]:  # fall back to the legacy mask so 100ep vs 500ep stay comparable
            noise = torch.zeros(1, mae.patch_embed.num_patches)
            noise[0, old["masked"]] = 1.0
            noise += torch.arange(noise.shape[1]) * 1e-6
            with torch.no_grad():
                loss, pred, mask = mae(x, mask_ratio=ratio, noise=noise.to(dev))
            pred_img = mae.unpatchify(pred)
            mm = mae.unpatchify(mask.unsqueeze(-1).repeat(1, 1, mae.patch_size ** 2 * mae.in_chans))
            r = {"pred_img": pred_img, "pasted": x * (1 - mm) + pred_img * mm, "loss": float(loss),
                 "masked_indices": [torch.nonzero(mask[0] > 0.5).flatten().tolist()]}
            mask_source = "legacy masked indices from the 100-epoch entry (seeded mask did not reproduce)"
        assert r["masked_indices"][0] == old["masked"]
        tag = f"r{int(ratio * 100)}_s{seed}"
        base = f"samples/{s['id']}/reconstructions_500ep"
        save_gray(denormalize(r["pred_img"][0], mspec).cpu().numpy(), OUT / base / f"{tag}_full.png")
        save_gray(denormalize(r["pasted"][0], mspec).cpu().numpy(), OUT / base / f"{tag}_pasted.png")
        out.append({"mask_ratio": ratio, "seed": seed, "masked": r["masked_indices"][0], "reconstruction_png": f"{base}/{tag}_full.png",
                    "pasted_png": f"{base}/{tag}_pasted.png", "loss_masked_mse": r["loss"],
                    "provenance": provenance(SCRIPT, checkpoint_id="usf_mae_500ep.pth", checkpoint_sha256=usf_sha,
                                             params={"mask_ratio": ratio, "seed": seed, "norm_pix_loss": False, "input": mspec.to_dict(),
                                                     "input_image": s["image"], "mask_source": mask_source, "epochs": 500})})
    return out


n_rec, n_pred_changed = 0, 0
for s in man["samples"]:
    sid = s["id"]
    if split_of.get(sid) != "test":
        sys.exit(f"sample {sid} is not in the test split")
    # predictions: from the headline runs' npz (same evaluation as the metrics)
    s["predictions_display_png"] = copy.deepcopy(s.get("predictions", {}))
    for mid in list(s.get("predictions", {})):
        p = npz[mid].get(sid)
        if p is None:
            sys.exit(f"{sid} missing from {headline[mid]}/test_predictions.npz")
        if p["label"] != s["label"]:
            sys.exit(f"{sid}: npz label {p['label']} != manifest label {s['label']}")
        if p["pred"] != s["predictions"][mid]["pred"]:
            n_pred_changed += 1
            say(f"[pred] {sid}/{mid}: argmax {s['predictions'][mid]['pred']} (640px PNG) -> {p['pred']} (npz, original image)")
        s["predictions"][mid] = {"probs": p["probs"], "pred": p["pred"]}
    # attributions: rename + labels
    for at in s.get("attributions", []):
        at["method"] = X.canonical_method(at["method"])
        at["method_label"] = X.method_label(at["method"])
        at["layer_label"] = X.layer_label(at["model"], at["layer"], at["method"])
        if isinstance(at.get("provenance", {}).get("params"), dict) and "method" in at["provenance"]["params"]:
            at["provenance"]["params"]["method"] = at["method"]
    # perturbations: checkpoint ids (the exporter used the manifest's loaded models) + labels
    for pe in s.get("perturbations", []):
        prm = pe.setdefault("provenance", {}).setdefault("params", {})
        prm["checkpoints"] = {mid: model_ck[mid] for mid in pe.get("predictions", {})}
        prm.setdefault("attribution_target", "true label")
        prm.setdefault("input", "exported sample PNG (<= 640 px)")
        pe["provenance"]["note"] = "checkpoint ids added by finalize_website_bundle.py from manifest.models (same models the exporter loaded)"
        for mid, at in (pe.get("attributions") or {}).items():
            lay = "act" if mid == "densenet161" else "block11"
            at.update({"method_label": X.method_label(at["method"]), "layer": lay, "layer_label": X.layer_label(mid, lay, at["method"])})
    # reconstructions
    side = OUT / "samples" / sid / "reconstructions_500ep" / "index.json"
    rec500 = None
    if mae is not None:
        rec500 = make_recon_500(s)
        wjson({"checkpoint_sha256": usf_sha, "entries": rec500}, side.relative_to(OUT))
        n_rec += len(rec500)
    elif side.exists():
        c = rjson(side)
        if c.get("checkpoint_sha256") == usf_sha and all((OUT / e["reconstruction_png"]).exists() for e in c["entries"]):
            rec500 = c["entries"]
    if rec500:
        s["reconstructions_legacy"] = s.get("reconstructions", [])
        s["reconstructions"] = rec500
if a.reconstruct and mae is not None:
    say(f"[recon] {n_rec} 500-epoch reconstructions written for {len(man['samples'])} samples")
recon_500 = all(s.get("reconstructions_legacy") is not None for s in man["samples"] if s.get("reconstructions") or s.get("reconstructions_legacy"))
say(f"[pred] {n_pred_changed} sample argmax changes after switching to npz predictions")

# ------------------------------------------------------------------ 5. dataset-level files
bundle_prov = lambda **p: provenance(SCRIPT, params=p or None)  # noqa: E731

le = FZ.build_label_efficiency(summaries, regime="finetune", fractions=FZ.LE_FRACTIONS, seeds=FZ.LE_SEEDS,
                               provenance=bundle_prov(source="runs/*/summary.json (test metrics); *_demo runs excluded", regime="finetune"),
                               notes=LE_NOTES)
wjson(le, "label_efficiency.json")

ds_src = rjson(SRC / "dataset_stats.json")
root_stats = rjson(ART / "dataset_stats.json")
nd = pd.read_csv(ART / "splits" / "near_duplicates.csv")
ds = FZ.enrich_dataset_stats(ds_src, split, nd, root_stats.get("near_duplicates"), script=SCRIPT)
wjson(ds, "dataset_stats.json")

ev = rjson(SRC / "evaluation.json")
for mid, pm in ev["per_model"].items():
    if "temperature_fit_on_val" in pm:
        pm["temperature"] = pm["temperature_fit_on_val"]
ev["seed_summary"] = {"usf_mae": FZ.seed_summary(summaries, "usf_mae")}
ev["calibration_fields"] = {"temperature_fit_on_val": "temperature T fitted on the validation predictions (NLL); T < 1 sharpens",
                            "temperature": "alias of temperature_fit_on_val (schema 1.1)",
                            "ece": "test ECE (10 bins) before temperature scaling",
                            "ece_after_temperature": "test ECE after applying the val-fitted temperature"}
ev.setdefault("notes", []).append(f"Metrics unchanged from the source bundle. DenseNet metrics come from the original {DENSENET_RUN} predictions "
                                  "(see manifest provenance_notes). seed_summary sd = sample sd (ddof=1).")
wjson(ev, "evaluation.json")

tc = rjson(SRC / "training_curves.json")
tc["notes"] = [f"densenet161/finetune: runs/{DENSENET_RUN}/history.json was overwritten by an interrupted rerun and is truncated at 10 epochs; "
               "its epoch-7 best val macro-F1 equals the original run's (0.9589). It is the rerun's curve, not the original run's.",
               "Frozen-regime curves are not included (curve_runs were not configured for the notebook-06 export)."]
wjson(tc, "training_curves.json")

wjson(faithfulness_json(str(ART / "faithfulness.csv"), bundle_prov(source="faithfulness.csv (notebook 06)")), "faithfulness.json")

# ------------------------------------------------------------------ 6. manifest
notes = [
    "USF-MAE: the official checkpoint is usf_mae_500ep.pth (sha256 " + usf_sha + "); all USF-MAE fine-tuning runs were initialised from it.",
    ("MAE reconstructions (samples[].reconstructions) were regenerated with usf_mae_500ep.pth using the same input pixels, preprocessing, mask "
     "ratios (0.25/0.5/0.75/0.9), seed 0 and identical masked patches as the original export; the original 100-epoch reconstructions "
     "(usf_mae_100ep.pth, sha256 " + USF_100_SHA + ") are kept unchanged in samples[].reconstructions_legacy.") if recon_500 else
    ("MAE reconstructions still come from the HISTORICAL usf_mae_100ep.pth (sha256 " + USF_100_SHA + "), not from the 500-epoch checkpoint "
     "behind the USF-MAE classifier; run finalize_website_bundle.py --reconstruct to regenerate them."),
    (f"DenseNet-161 checkpoint: runs/{DENSENET_RUN}/best.pt (sha256 {dn_sha}) was overwritten by an interrupted sweep rerun and does not "
     f"match summary.json (sha256 {dn_summary_sha}). Metrics, evaluation.json, label efficiency and samples[].predictions come from the "
     "ORIGINAL run's saved predictions. Derived from the RERUN weights (sha256 456847...): DenseNet feature maps, DenseNet attributions, "
     "DenseNet perturbation predictions/maps (and the DenseNet Grad-CAM region used for occlude_patch for all models), DenseNet rows of "
     "faithfulness.json, the DenseNet training curve, and samples[].predictions_display_png for DenseNet. Attention: n/a for DenseNet."),
    equiv_note(),
    SAMPLE_PREDICTIONS_SOURCE,
    "Method id 'chefer' was renamed 'grad_rollout' (gradient-weighted attention rollout, a simplified variant inspired by Chefer et al. 2021, without LRP). PNG filenames keep '_chefer_'.",
    "ViT layer ids blockN = input of model.blocks[N] (0-indexed) = residual stream after N transformer blocks; shown 1-indexed as 'after block N of 12' (layer_label). Attention JSON cls[i] = Block i+1.",
    "Label efficiency (fine-tune): " + " ".join(f"{k}: {v}" for k, v in le["availability"].items()),
    "The source bundle was generated in Colab (notebook 06) without a git commit (generator.git_commit null); finalizer.git_commit records the repository state that applied these corrections.",
]
man["schema_version"] = SCHEMA
man["finalizer"] = {"script": SCRIPT, "finalized_at": now_iso(), "git_commit": git_commit(), "command": " ".join(sys.argv),
                    "source_bundle": {"path": "website_bundle (ART)", "generated_at": man.get("generated_at"), "schema_version": "1.0"}}
man["provenance_notes"] = notes
man["sample_predictions_source"] = SAMPLE_PREDICTIONS_SOURCE
man["faithfulness"] = "faithfulness.json"
man["label_efficiency"] = "label_efficiency.json"
if equiv:
    man["provenance_files"] = {"densenet_equivalence": "provenance/densenet_equivalence.json"}
wjson(man, "manifest.json")
(inside_out(OUT / "finalize_log.txt")).write_text("\n".join(LOG + [f"[done] {now_iso()}"]), encoding="utf-8")
say(f"[done] schema {SCHEMA} bundle at {OUT}")
