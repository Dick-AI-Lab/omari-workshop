"""Functional-equivalence check: does a run's current best.pt reproduce that run's saved {val,test}_predictions.npz?

Used to qualify runs/densenet161_finetune_lf1_s0/best.pt, which was overwritten by an interrupted rerun (its sha256 no
longer matches summary.json). A self-consistent run (e.g. vit_b16_finetune_lf1_s0) is the positive control that
calibrates the expected numerical noise (different GPU / torch version / fp16 kernels). Inference only; read-only.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, Sequence

import numpy as np
import pandas as pd


def functional_equivalence(artifacts: str | os.PathLike, images: str | os.PathLike, runs: Sequence[str], device: str = "cuda",
                           batch_size: int = 64) -> Dict:
    import torch
    import torch.nn.functional as F
    from torch.utils.data import DataLoader

    from ..datasets import FetalPlanesDataset, build_transforms, input_spec
    from ..evaluation.metrics import classification_report
    from ..models import load_finetuned
    from ..utils import now_iso, sha256_file

    art = Path(artifacts)
    dev = torch.device(device)
    man = pd.read_csv(art / "splits" / "split_manifest.csv")
    man["path"] = [os.path.join(str(images), os.path.basename(p)) for p in man["path"]]

    @torch.no_grad()
    def predict(model, df, spec, amp):
        dl = DataLoader(FetalPlanesDataset(df, build_transforms(spec, False)), batch_size=batch_size, num_workers=0)
        ps = []
        for x, _, _ in dl:
            with torch.autocast(device_type=dev.type, enabled=amp and dev.type == "cuda"):
                lo = model(x.to(dev))
            ps.append(F.softmax(lo.float(), -1).cpu().numpy())
        return np.concatenate(ps)

    res = {"generated_at": now_iso(),
           "method": ("Load best.pt with omari_us.models.load_finetuned, eval mode; run on the exact val/test images of the run's "
                      "{val,test}_predictions.npz (ids looked up in splits/split_manifest.csv, paths re-rooted to the local Images folder); "
                      "preprocessing = the run's summary.json 'input' spec (build_transforms train=False); softmax in fp32; compare with the "
                      "saved npz probabilities by image id. AMP off is the primary fp32 comparison; AMP on matches how the original "
                      "predictions were written (amp=true, Colab T4, torch 2.14.1)."),
           "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if dev.type == "cuda" else str(dev)}, "runs": {}}
    for run in runs:
        rd = art / "runs" / run
        with open(rd / "summary.json", encoding="utf-8") as f:
            summ = json.load(f)
        mid = summ["config"]["model"]
        model, meta = load_finetuned(mid, str(rd / "best.pt"))
        model = model.to(dev).eval()
        spec = input_spec(mid, {k: v for k, v in summ["input"].items() if k in ("size", "resize", "interpolation", "mean", "std", "channels")})
        sha = sha256_file(rd / "best.pt")
        r = {"best_pt_sha256": sha, "summary_checkpoint_sha256": summ["checkpoint_sha256"], "sha_matches_summary": sha == summ["checkpoint_sha256"],
             "best_pt_meta_epoch": meta.get("epoch"), "best_pt_meta_val_macro_f1": meta.get("val_macro_f1"), "best_pt_meta_created": meta.get("created"),
             "summary_best_epoch": summ["best_epoch"], "summary_best_val_macro_f1": summ["best_val_macro_f1"], "input_spec": spec.to_dict(), "splits": {}}
        for split in ("val", "test"):
            z = np.load(rd / f"{split}_predictions.npz", allow_pickle=True)
            ids = [str(i) for i in z["ids"]]
            df = man.set_index("Image_name").loc[ids].reset_index()
            assert (df["split"] == split).all()
            y = z["labels"].astype(int)
            assert (df["label"].to_numpy() == y).all()
            orig = z["probs"].astype(np.float64)
            o = classification_report(y, orig.argmax(1))
            sr = {"n": len(ids), "original_npz": {"accuracy": float(o["accuracy"]), "macro_f1": float(o["macro_f1"])}}
            for amp in (False, True):
                p = predict(model, df, spec, amp).astype(np.float64)
                d = np.abs(p - orig)
                rep = classification_report(y, p.argmax(1))
                sr["amp_on" if amp else "amp_off"] = {
                    "max_abs_dprob": float(d.max()), "mean_abs_dprob": float(d.mean()), "p99_abs_dprob": float(np.quantile(d.max(1), 0.99)),
                    "argmax_agree": int((p.argmax(1) == orig.argmax(1)).sum()), "argmax_disagree": int((p.argmax(1) != orig.argmax(1)).sum()),
                    "recomputed_accuracy": float(rep["accuracy"]), "recomputed_macro_f1": float(rep["macro_f1"])}
            r["splits"][split] = sr
        res["runs"][run] = r
        del model
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    return res
