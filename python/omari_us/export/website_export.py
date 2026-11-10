"""Write the website artifact bundle (schema v1; see docs/ARTIFACT_SCHEMA.md and website/src/data/types.ts).

Everything written here comes from real inference with the checkpoints named in the config. Each item carries
provenance (script, checkpoint id + sha256, parameters, time, git commit). Nothing is interpolated or invented:
if a checkpoint or input is missing, the corresponding section is simply omitted and the website shows its
empty state.

Usage:
    python scripts/export_website_artifacts.py --config configs/export.yaml
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image

from .. import explainability as X
from ..datasets import build_transforms, input_spec
from ..datasets import perturbations as P
from ..datasets.fetal_planes import dataset_stats
from ..evaluation import metrics as M
from ..models import load_finetuned
from ..utils import CLASSES, MODEL_IDS, NUM_CLASSES, get_device, git_commit, now_iso, provenance, sha256_file, write_json

SCHEMA_VERSION = "1.0"
SCRIPT = "omari_us.export.website_export"
DISPLAY = {"densenet161": "DenseNet-161", "vit_b16": "ViT-B/16", "usf_mae": "USF-MAE"}


@dataclass
class LoadedModel:
    id: str
    model: torch.nn.Module
    meta: Dict
    checkpoint: str
    sha256: str
    transform: object
    spec: object

    @property
    def ckpt_id(self) -> str:
        return f"{os.path.basename(os.path.dirname(self.checkpoint))}/{os.path.basename(self.checkpoint)}"

    def prov(self, **params):
        return provenance(SCRIPT, checkpoint_id=self.ckpt_id, checkpoint_sha256=self.sha256, params=params or None)


# ------------------------------------------------------------------------------------------------ io helpers
def save_gray_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.clip(arr, 0, 1) * 255).round().astype(np.uint8), mode="L").save(path, optimize=True)


def save_map_png(m: torch.Tensor, path: Path) -> Dict[str, float]:
    a = m.detach().float().cpu().numpy()
    lo, hi = float(a.min()), float(a.max())
    save_gray_png((a - lo) / (hi - lo) if hi > lo else np.zeros_like(a), path)
    return {"raw_min": lo, "raw_max": hi, "resolution": [int(a.shape[0]), int(a.shape[1])]}


def load_gray(path: str, max_side: int) -> np.ndarray:
    with Image.open(path) as im:
        im = im.convert("L")
        s = min(1.0, max_side / max(im.size))
        if s < 1:
            im = im.resize((round(im.size[0] * s), round(im.size[1] * s)), Image.LANCZOS)
        return np.asarray(im, dtype=np.float32) / 255.0


def to_input(lm: LoadedModel, gray: np.ndarray, device) -> torch.Tensor:
    pil = Image.fromarray((np.clip(gray, 0, 1) * 255).round().astype(np.uint8), mode="L")
    return lm.transform(pil).unsqueeze(0).to(device)


@torch.no_grad()
def predict_one(lm: LoadedModel, gray: np.ndarray, device) -> Dict:
    p = F.softmax(lm.model(to_input(lm, gray, device)).float(), -1)[0].cpu().numpy()
    return {"probs": [round(float(v), 6) for v in p], "pred": int(p.argmax())}


# ------------------------------------------------------------------------------------------------ main class
class Exporter:
    def __init__(self, cfg: Dict, root: str = "."):
        self.cfg = cfg
        self.root = Path(root)
        self.out = Path(cfg.get("output_dir", "website/public/experiments"))
        if not self.out.is_absolute():
            self.out = self.root / self.out
        self.device = get_device(cfg.get("device"))
        self.max_side = int(cfg.get("max_side", 640))
        self.models: Dict[str, LoadedModel] = {}
        self.log: List[str] = []

    def say(self, s: str):
        print(s)
        self.log.append(s)

    # -- setup
    def load_models(self):
        for mid in MODEL_IDS:
            mc = (self.cfg.get("models") or {}).get(mid) or {}
            ck = mc.get("checkpoint")
            if not ck or not os.path.exists(ck):
                self.say(f"[skip] {mid}: fine-tuned checkpoint not found ({ck})")
                continue
            model, meta = load_finetuned(mid, ck, map_location="cpu")
            spec = input_spec(mid, meta.get("input") and {k: v for k, v in meta["input"].items() if k in ("size", "resize", "interpolation", "mean", "std", "channels")})
            self.models[mid] = LoadedModel(mid, model.to(self.device).eval(), meta, ck, sha256_file(ck), build_transforms(spec, train=False), spec)
            self.say(f"[ok] {mid}: {ck} (init: {meta.get('init_source')})")

    def model_info(self) -> List[Dict]:
        out = []
        for mid, lm in self.models.items():
            c = lm.meta.get("config", {})
            n_params = sum(p.numel() for p in lm.model.parameters())
            ft = f"{c.get('regime', '?')}; {c.get('epochs', '?')} epochs max; lr {c.get('lr')}; loss {c.get('loss')}; label fraction {c.get('label_fraction')}; seed {c.get('seed')}; best epoch {lm.meta.get('epoch')}"
            out.append({"id": mid, "display_name": DISPLAY[mid], "checkpoint_id": lm.ckpt_id, "checkpoint_sha256": lm.sha256,
                        "initialization": lm.meta.get("init_source", "unknown"), "finetuning": ft, "params": int(n_params),
                        "input": lm.spec.to_dict(), "notes": (self.cfg.get("models") or {}).get(mid, {}).get("notes")})
        return out

    # -- sample selection uses saved test predictions when available (no extra inference)
    def select_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        sc = self.cfg.get("samples", {})
        split = sc.get("split", "test")
        pool = df[df["split"] == split].reset_index(drop=True)
        rng = np.random.default_rng(sc.get("seed", 0))
        chosen: Dict[str, List[str]] = {}

        def add(name, tag):
            chosen.setdefault(name, [])
            if tag not in chosen[name]:
                chosen[name].append(tag)

        for c in range(NUM_CLASSES):
            names = pool.loc[pool["label"] == c, "Image_name"].to_numpy()
            for n in rng.choice(names, size=min(sc.get("per_class", 2), len(names)), replace=False):
                add(n, f"class-{c}")
        preds = self._load_run_predictions()
        if preds:
            common = set.intersection(*[set(p["ids"]) for p in preds.values()]) if len(preds) > 1 else set(next(iter(preds.values()))["ids"])
            names = sorted(common)
            table = {mid: dict(zip(p["ids"], p["probs"].argmax(1))) for mid, p in preds.items()}
            labels = dict(zip(next(iter(preds.values()))["ids"], next(iter(preds.values()))["labels"]))
            dis = [n for n in names if len({table[m][n] for m in table}) > 1]
            for n in rng.permutation(dis)[: sc.get("disagreements", 6)]:
                add(str(n), "disagreement")
            for mid in table:
                wrong = [n for n in names if table[mid][n] != labels[n]]
                for n in rng.permutation(wrong)[: sc.get("misclassified_per_model", 2)] if wrong else []:
                    add(str(n), f"misclassified-{mid}")
        opening = sc.get("opening")
        if opening:
            add(opening, "opening")
        elif chosen:
            add(next(iter(chosen)), "opening")
        if sc.get("max_samples"):
            keep = [n for n in chosen if "opening" in chosen[n]] + [n for n in chosen if "opening" not in chosen[n]]
            chosen = {n: chosen[n] for n in keep[: int(sc["max_samples"])]}
        sel = df[df["Image_name"].isin(chosen)].copy()
        sel["tags"] = sel["Image_name"].map(lambda n: chosen[n])
        self.say(f"[samples] {len(sel)} selected")
        return sel

    def _load_run_predictions(self) -> Dict[str, Dict]:
        out = {}
        for mid, run in (self.cfg.get("evaluation", {}).get("runs") or {}).items():
            f = os.path.join(run, "test_predictions.npz")
            if os.path.exists(f):
                z = np.load(f, allow_pickle=True)
                out[mid] = {k: z[k] for k in z.files}
        return out

    # -- per-sample artifacts
    def export_sample(self, r: pd.Series) -> Dict:
        sid = r["Image_name"]
        sdir = self.out / "samples" / sid
        gray = load_gray(r["path"], self.max_side)
        save_gray_png(gray, sdir / "image.png")
        s: Dict = {"id": sid, "image": f"samples/{sid}/image.png", "width": int(gray.shape[1]), "height": int(gray.shape[0]),
                   "label": int(r["label"]), "split": r["split"], "patient": f"Patient{int(r['Patient_num']):05d}", "tags": list(r["tags"]),
                   "predictions": {}}
        for k in ("Brain_plane", "US_Machine", "Operator"):
            if k in r and isinstance(r[k], str):
                s[{"Brain_plane": "brain_plane", "US_Machine": "machine", "Operator": "operator"}[k]] = r[k]
        for mid, lm in self.models.items():
            s["predictions"][mid] = predict_one(lm, gray, self.device)
        sec = self.cfg
        if sec.get("features", {}).get("enabled", True) and "densenet161" in self.models:
            s["features"] = {"densenet161": self._features(self.models["densenet161"], gray, sdir)}
        att_cfg = sec.get("attention", {})
        if att_cfg.get("enabled", True):
            full_for = att_cfg.get("full_for", "opening")
            want_full = full_for == "all" or (full_for == "opening" and "opening" in s["tags"])
            s["attention"] = {mid: self._attention(lm, gray, sdir, want_full, att_cfg.get("blocks_full", [0, 5, 11])) for mid, lm in self.models.items() if mid != "densenet161"}
        if sec.get("attributions", {}).get("enabled", True):
            s["attributions"] = self._attributions(gray, s, sdir)
        if sec.get("reconstructions", {}).get("enabled", True):
            rec = self._reconstructions(gray, sdir, sid)
            if rec:
                s["reconstructions"] = rec
        if sec.get("perturbations", {}).get("enabled", True) and self.models:
            s["perturbations"] = self._perturbations(gray, s, sdir)
        return s

    def _features(self, lm: LoadedModel, gray, sdir) -> List[Dict]:
        layers = self.cfg.get("features", {}).get("layers", ["features.denseblock1", "features.denseblock2", "features.denseblock3", "features.denseblock4"])
        k = int(self.cfg.get("features", {}).get("channels", 12))
        acts = {}
        hs = []
        for name in layers:
            mod = lm.model
            for part in name.split("."):
                mod = getattr(mod, part)
            hs.append(mod.register_forward_hook(lambda m, i, o, name=name: acts.__setitem__(name, o.detach().cpu()[0])))
        with torch.no_grad():
            lm.model(to_input(lm, gray, self.device))
        for h in hs:
            h.remove()
        out = []
        for name in layers:
            a = torch.relu(acts[name])
            means = a.mean(dim=(1, 2))
            top = torch.argsort(means, descending=True)[:k].tolist()
            chans = []
            for c in top:
                ch = a[c]
                save_gray_png(((ch - ch.min()) / (ch.max() - ch.min() + 1e-12)).numpy(), sdir / "features" / f"{name}_{c}.png")
                chans.append({"index": int(c), "png": f"samples/{sdir.name}/features/{name}_{c}.png", "min": float(ch.min()), "max": float(ch.max()), "mean": float(ch.mean())})
            out.append({"layer": name, "shape": list(a.shape), "channels": chans, "selection": f"top {k} channels by mean ReLU activation on this image"})
        return out

    def _attention(self, lm: LoadedModel, gray, sdir, want_full: bool, blocks_full) -> Dict:
        x = to_input(lm, gray, self.device)
        A = X.vitx.attention_maps(lm.model, x).cpu()  # (L, H, T, T)
        L, H, T, _ = A.shape
        cls = A[:, :, 0, 1:]
        roll = X.vitx.attention_rollout(lm.model, x).flatten()
        f = sdir / "attention" / f"{lm.id}_cls.json"
        write_json({"blocks": L, "heads": H, "grid": int(round((T - 1) ** 0.5)), "cls": np.round(cls.numpy(), 6).tolist(),
                    "rollout": np.round(roll.numpy(), 6).tolist(), "provenance": lm.prov(kind="attention")}, f)
        entry: Dict = {"cls": f"samples/{sdir.name}/attention/{lm.id}_cls.json"}
        if want_full:
            entry["full"] = []
            for b in blocks_full:
                b = int(b) % L
                arr = np.clip(np.round(A[b].numpy() * 65535), 0, 65535).astype("<u2")
                p = sdir / "attention" / f"{lm.id}_block{b}.u16.bin"
                p.parent.mkdir(parents=True, exist_ok=True)
                arr.tofile(p)
                entry["full"].append({"block": b, "path": f"samples/{sdir.name}/attention/{p.name}", "heads": H, "tokens": T})
        return entry

    def _attributions(self, gray, s, sdir) -> List[Dict]:
        ac = self.cfg.get("attributions", {})
        methods_cfg = ac.get("methods", "all")
        out = []
        for mid, lm in self.models.items():
            x = to_input(lm, gray, self.device)
            targets = {s["label"], s["predictions"][mid]["pred"]}
            methods = X.methods_for(mid) if methods_cfg == "all" else [m for m in methods_cfg if X.compatible(m, mid)]
            for meth in methods:
                layers = X.layers_for(meth, mid) if ac.get("all_layers", True) else (X.layers_for(meth, mid)[0],)
                for t in sorted(targets) if X.REGISTRY[meth].class_specific else [s["label"]]:
                    for layer in layers:
                        try:
                            m = X.explain(meth, mid, lm.model, x, t, layer)
                        except X.IncompatibleMethod:
                            continue
                        name = f"{mid}_{meth}_{layer or 'final'}_c{t}.png"
                        info = save_map_png(m, sdir / "attributions" / name)
                        out.append({"model": mid, "method": meth, "layer": layer or "final", "target_class": int(t), "png": f"samples/{sdir.name}/attributions/{name}",
                                    **info, "provenance": lm.prov(method=meth, layer=layer or "final", target_class=int(t))})
        return out

    def _reconstructions(self, gray, sdir, sid) -> List[Dict]:
        rc = self.cfg.get("reconstructions", {})
        ck = (self.cfg.get("models") or {}).get("usf_mae", {}).get("pretrain_checkpoint")
        if not ck or not os.path.exists(ck):
            return []
        if not hasattr(self, "_mae"):
            from ..models.usf_mae import build_mae_from_state_dict, load_usf_mae_checkpoint

            sd, rep = load_usf_mae_checkpoint(ck)
            if not rep.ok_for_reconstruction:
                self.say("[skip] reconstructions: checkpoint has no complete decoder\n" + rep.summary())
                self._mae = None
            else:
                self._mae = build_mae_from_state_dict(sd).to(self.device)
                self._mae_rep = rep
                self._mae_spec = input_spec("usf_mae")
                self._mae_tf = build_transforms(self._mae_spec, train=False)
        if self._mae is None:
            return []
        from ..datasets.transforms import denormalize
        from ..models.usf_mae import reconstruct

        pil = Image.fromarray((gray * 255).round().astype(np.uint8), mode="L")
        x = self._mae_tf(pil).unsqueeze(0).to(self.device)
        out = []
        for ratio in rc.get("ratios", [0.25, 0.5, 0.75, 0.9]):
            for seed in rc.get("seeds", [0]):
                r = reconstruct(self._mae, x, float(ratio), int(seed))
                rec = denormalize(r["pred_img"][0], self._mae_spec).cpu().numpy()
                pas = denormalize(r["pasted"][0], self._mae_spec).cpu().numpy()
                tag = f"r{int(ratio * 100)}_s{seed}"
                save_gray_png(rec, sdir / "reconstructions" / f"{tag}_full.png")
                save_gray_png(pas, sdir / "reconstructions" / f"{tag}_pasted.png")
                out.append({"mask_ratio": float(ratio), "seed": int(seed), "masked": r["masked_indices"][0],
                            "reconstruction_png": f"samples/{sid}/reconstructions/{tag}_full.png", "pasted_png": f"samples/{sid}/reconstructions/{tag}_pasted.png",
                            "loss_masked_mse": r["loss"],
                            "provenance": provenance(SCRIPT, checkpoint_id=os.path.basename(ck), checkpoint_sha256=self._mae_rep.sha256,
                                                     params={"mask_ratio": ratio, "seed": seed, "norm_pix_loss": False, "input": self._mae_spec.to_dict()})})
        return out

    def _perturbations(self, gray, s, sdir) -> List[Dict]:
        pc = self.cfg.get("perturbations", {})
        grid = [tuple(g) for g in pc.get("grid", P.DEFAULT_GRID)]
        out = []
        # occlusion of the region DenseNet Grad-CAM ranks highest (top 10% of pixels), if DenseNet is available
        if "densenet161" in self.models and pc.get("occlude_top", True):
            lm = self.models["densenet161"]
            m = X.explain("gradcam", "densenet161", lm.model, to_input(lm, gray, self.device), s["label"])
            up = F.interpolate(m[None, None], size=gray.shape, mode="bilinear", align_corners=False)[0, 0].numpy()
            region = up >= np.quantile(up, 0.9)
            grid.append(("occlude_patch", 0.1, "top-10% Grad-CAM (DenseNet) region blurred", region))
        for item in grid:
            kind, param, label = item[0], item[1], item[2]
            region = item[3] if len(item) > 3 else None
            img = P.apply(gray, kind, param, seed=int(pc.get("seed", 0)), region=region)
            tag = f"{kind}_{str(param).replace('.', 'p').replace('-', 'm')}"
            save_gray_png(img, sdir / "perturbations" / f"{tag}.png")
            e = {"type": kind, "param": float(param), "label": label, "image_png": f"samples/{sdir.name}/perturbations/{tag}.png", "predictions": {}, "attributions": {},
                 "provenance": provenance(SCRIPT, params={"type": kind, "param": float(param), "seed": int(pc.get("seed", 0)), "semantics": "omari_us.datasets.perturbations"})}
            for mid, lm in self.models.items():
                e["predictions"][mid] = predict_one(lm, img, self.device)
                meth = "gradcam" if mid == "densenet161" else "vit_gradcam"
                mm = X.explain(meth, mid, lm.model, to_input(lm, img, self.device), s["label"])
                name = f"{tag}_{mid}_{meth}.png"
                save_map_png(mm, sdir / "perturbations" / name)
                e["attributions"][mid] = {"method": meth, "png": f"samples/{sdir.name}/perturbations/{name}"}
            out.append(e)
        return out

    # -- dataset-level artifacts
    def export_evaluation(self, df: pd.DataFrame) -> Optional[str]:
        ec = self.cfg.get("evaluation", {})
        preds = self._load_run_predictions()
        if not preds:
            self.say("[skip] evaluation: no test_predictions.npz found")
            return None
        per_model, probs_out = {}, {}
        ids_ref, labels_ref, machines_ref = None, None, None
        for mid, z in preds.items():
            y, p = z["labels"].astype(int), z["probs"]
            rep = M.classification_report(y, p.argmax(1))
            ci = M.bootstrap_ci(y, p.argmax(1), groups=z.get("patients"), n_boot=int(ec.get("bootstrap", 1000)))
            rel = M.reliability(p, y)
            rep.update({"ci95": {"accuracy": ci["accuracy"], "macro_f1": ci["macro_f1"]}, "ece": rel["ece"], "nll": M.nll(p, y)})
            if "machines" in z and len(set(z["machines"])) > 1:
                rep["by_machine"] = M.by_group(y, p.argmax(1), z["machines"])
            val = os.path.join(ec["runs"][mid], "val_predictions.npz")
            if os.path.exists(val):
                v = np.load(val, allow_pickle=True)
                t = M.fit_temperature(v["probs"], v["labels"].astype(int))
                rep["temperature_fit_on_val"] = t
                rep["ece_after_temperature"] = M.reliability(M.apply_temperature(p, t), y)["ece"]
            per_model[mid] = rep
            order = np.argsort(z["ids"])
            probs_out[mid] = np.round(p[order], 4).tolist()
            if ids_ref is None:
                ids_ref, labels_ref = z["ids"][order], y[order]
                machines_ref = z["machines"][order] if "machines" in z else None
        comps = []
        mids = list(preds)
        for i in range(len(mids)):
            for j in range(i + 1, len(mids)):
                a, b = preds[mids[i]], preds[mids[j]]
                if list(np.sort(a["ids"])) != list(np.sort(b["ids"])):
                    continue
                oa, ob = np.argsort(a["ids"]), np.argsort(b["ids"])
                t = M.mcnemar(a["labels"][oa].astype(int), a["probs"][oa].argmax(1), b["probs"][ob].argmax(1))
                comps.append({"a": mids[i], "b": mids[j], "test": "exact McNemar", "statistic": t["statistic"], "p_value": t["p_value"]})
        if comps:
            for c, adj in zip(comps, M.holm([c["p_value"] for c in comps])):
                c["note"] = f"Holm-adjusted p = {adj:.3g}; image-level pairing ignores patient clustering"
        ev = {"provenance": provenance(SCRIPT, params={"bootstrap": int(ec.get("bootstrap", 1000)), "cluster": "patient"}), "split": "test",
              "n": int(len(labels_ref)), "per_model": per_model, "comparisons": comps}
        if ec.get("export_test_probabilities", True):
            ev["test_predictions"] = {"ids": [str(i) for i in ids_ref], "labels": [int(v) for v in labels_ref], "probs": probs_out}
            if machines_ref is not None:
                ev["test_predictions"]["machines"] = [str(m) for m in machines_ref]
        # misclassified thumbnails
        k = int(ec.get("max_misclassified", 24))
        mis = []
        name_to_path = dict(zip(df["Image_name"], df["path"]))
        for mid, z in preds.items():
            wrong = [str(n) for n, yy, pp in zip(z["ids"], z["labels"], z["probs"].argmax(1)) if yy != pp][:k]
            for n in wrong:
                if n in name_to_path and not any(m["id"] == n for m in mis):
                    save_gray_png(load_gray(name_to_path[n], 256), self.out / "misclassified" / f"{n}.png")
                    preds_n = {m2: int(dict(zip(preds[m2]["ids"], preds[m2]["probs"].argmax(1))).get(n, -1)) for m2 in preds}
                    mis.append({"id": n, "image": f"misclassified/{n}.png", "label": int(df.loc[df["Image_name"] == n, "label"].iloc[0]), "preds": preds_n})
        ev["misclassified"] = mis
        write_json(ev, self.out / "evaluation.json")
        return "evaluation.json"

    def export_curves(self) -> Optional[str]:
        """Training curves from history.json. evaluation.curve_runs: {model: {regime: run_dir}}; defaults to the main runs."""
        ec = self.cfg.get("evaluation", {})
        curve_runs = ec.get("curve_runs") or {}
        if not curve_runs:
            for mid, run in (ec.get("runs") or {}).items():
                summ = os.path.join(run, "summary.json")
                regime = json.load(open(summ))["config"]["regime"] if os.path.exists(summ) else "run"
                curve_runs[mid] = {regime: run}
        per = {}
        for mid, regimes in curve_runs.items():
            for regime, run in regimes.items():
                h = os.path.join(run, "history.json")
                if os.path.exists(h):
                    hist = json.load(open(h))
                    per.setdefault(mid, {})[regime] = {k: hist[k] for k in ("epoch", "train_loss", "val_loss", "val_macro_f1", "lr") if k in hist}
        if not per:
            return None
        write_json({"provenance": provenance(SCRIPT, params={"runs": curve_runs}), "per_model": per}, self.out / "training_curves.json")
        return "training_curves.json"

    def export_label_efficiency(self) -> Optional[str]:
        pat = self.cfg.get("evaluation", {}).get("label_efficiency_glob")
        if not pat:
            return None
        rows = []
        for f in glob.glob(pat):
            s = json.load(open(f))
            c = s["config"]
            rows.append((c["model"], c["regime"], float(c["label_fraction"]), int(c["seed"]), s["test"]["macro_f1"]))
        if not rows:
            return None
        df = pd.DataFrame(rows, columns=["model", "regime", "fraction", "seed", "macro_f1"])
        regime = self.cfg.get("evaluation", {}).get("label_efficiency_regime", df["regime"].mode()[0])
        df = df[df["regime"] == regime]
        fr = sorted(df["fraction"].unique())
        seeds = sorted(df["seed"].unique())
        per = {}
        for mid, g in df.groupby("model"):
            per[mid] = {"macro_f1": [[float(g[(g.fraction == f) & (g.seed == s)]["macro_f1"].iloc[0]) if len(g[(g.fraction == f) & (g.seed == s)]) else float("nan") for s in seeds] for f in fr]}
        write_json({"provenance": provenance(SCRIPT, params={"glob": pat}), "fractions": fr, "seeds": seeds, "regime": regime, "per_model": per}, self.out / "label_efficiency.json")
        return "label_efficiency.json"

    # -- orchestrate
    def run(self, df: pd.DataFrame, clean: bool = False) -> Dict:
        if clean and self.out.exists():
            shutil.rmtree(self.out)
        self.out.mkdir(parents=True, exist_ok=True)
        self.load_models()
        write_json(dataset_stats(df, script=SCRIPT), self.out / "dataset_stats.json")
        samples = []
        if self.models:
            sel = self.select_samples(df)
            try:
                from tqdm.auto import tqdm

                it = tqdm(list(sel.iterrows()), desc="samples")
            except ImportError:
                it = list(sel.iterrows())
            for _, r in it:
                samples.append(self.export_sample(r))
        manifest = {
            "schema_version": SCHEMA_VERSION,
            "bundle_id": self.cfg.get("bundle_id", "omari-fpdb"),
            "generated_at": now_iso(),
            "generator": {"script": SCRIPT, "version": "0.1.0", "git_commit": git_commit(), "command": self.cfg.get("_command")},
            "dataset": {"name": "FETAL_PLANES_DB", "license": "CC BY 4.0", "source_url": "https://zenodo.org/records/3904280",
                        "citation": "Burgos-Artizzu et al., Sci. Rep. 10:10200 (2020)",
                        "split_protocol": "Authors' patient-level train/test split; validation = patient-disjoint 10% of training (StratifiedGroupKFold)"},
            "classes": CLASSES,
            "models": self.model_info(),
            "samples": samples,
            "dataset_stats": "dataset_stats.json",
        }
        for key, fn in (("evaluation", lambda: self.export_evaluation(df)), ("training_curves", self.export_curves), ("label_efficiency", self.export_label_efficiency)):
            v = fn()
            if v:
                manifest[key] = v
        write_json(manifest, self.out / "manifest.json")
        (self.out / "export_log.txt").write_text("\n".join(self.log))
        self.say(f"[done] bundle written to {self.out}")
        return manifest
