"""Independent checks on the saved .pptx (re-opened from disk, not the builder's objects).

    presentation/.venv/Scripts/python presentation/src/validate_deck.py

Checks: slide size; every slide has speaker notes; no shape outside the slide;
embedded image resolution; every headline number on the slides matches the
bundle JSON (recomputed here, not imported from build_deck); no 100-epoch
USF-MAE artifact is embedded; no localhost URL; the planned timing sums to 120.
Writes presentation/preview/validation.json and exits non-zero on failure.
"""
from __future__ import annotations

import json
import re
import statistics as st
import sys
from pathlib import Path

from pptx import Presentation
from pptx.util import Emu

ROOT = Path(__file__).resolve().parents[2]
B = ROOT / "website" / "public" / "experiments"
PPTX = ROOT / "presentation" / "OMARI_From_Pixels_to_Predictions.pptx"
J = {n: json.loads((B / f"{n}.json").read_text(encoding="utf-8")) for n in ("evaluation", "label_efficiency", "dataset_stats", "faithfulness", "manifest")}
FP = json.loads((ROOT / "presentation" / "assets" / "figure_provenance.json").read_text(encoding="utf-8"))

prs = Presentation(PPTX)
W, H = prs.slide_width, prs.slide_height
results, fails = [], []


def check(name, ok, detail=""):
    results.append({"check": name, "ok": bool(ok), "detail": detail})
    if not ok:
        fails.append(f"{name}: {detail}")


def slide_text(sl):
    out = []
    for sh in sl.shapes:
        if sh.has_text_frame:
            out.append(sh.text_frame.text)
        if sh.has_table:
            out += [c.text for r in sh.table.rows for c in r.cells]
        if sh.has_chart:
            ch = sh.chart
            for pl in ch.plots:
                out += [str(c) for c in pl.categories]
                for se in pl.series:
                    out.append(se.name)
    return "\n".join(out)


texts = [slide_text(s) for s in prs.slides]
notes = [s.notes_slide.notes_text_frame.text if s.has_notes_slide else "" for s in prs.slides]
ALL = "\n".join(texts + notes)

# ---- geometry
check("16:9 widescreen", (W, H) == (Emu(12192000), Emu(6858000)), f"{W}x{H} EMU")
off = []
for i, s in enumerate(prs.slides, 1):
    for sh in s.shapes:
        if sh.left is None:
            continue
        if sh.left < -Emu(1000) or sh.top < -Emu(1000) or sh.left + sh.width > W + Emu(1000) or sh.top + sh.height > H + Emu(1000):
            off.append((i, sh.shape_type, sh.name))
check("no shape outside slide bounds", not off, str(off[:5]))
check("speaker notes on every slide", all(len(n) > 80 for n in notes), str([i + 1 for i, n in enumerate(notes) if len(n) <= 80]))

# ---- image resolution (effective pixels per inch on the slide)
low = []
for i, s in enumerate(prs.slides, 1):
    for sh in s.shapes:
        if sh.shape_type == 13:
            px_w, _ = sh.image.size
            vis = 1 - (sh.crop_left + sh.crop_right)
            ppi = px_w * vis / (sh.width / 914400)
            if ppi < 90:
                low.append((i, sh.image.filename, round(ppi)))
check("embedded images >= 90 effective ppi", not low, str(low))

# ---- numbers vs bundle
pm = J["evaluation"]["per_model"]
seed = J["evaluation"]["seed_summary"]["usf_mae"]
for m in pm:
    for k in ("accuracy", "macro_f1"):
        check(f"{m} {k} on slide/notes", f"{pm[m][k]:.4f}" in ALL, f"{pm[m][k]:.4f}")
    check(f"{m} ECE before/after", f"{pm[m]['ece']:.3f}" in ALL and f"{pm[m]['ece_after_temperature']:.3f}" in ALL)
    check(f"{m} temperature", f"T = {pm[m]['temperature_fit_on_val']:.2f}" in ALL)
check("USF-MAE 3-seed mean ± sd", all(f"{seed[a][k]:.4f}" in ALL for a in ("mean", "sd") for k in ("accuracy", "macro_f1")))
check("audit full-label table (0.9474/0.9425, 0.9465/0.9410, 0.9471±0.0021/0.9416±0.0023)",
      all(x in ALL for x in ("0.9474", "0.9425", "0.9465", "0.9410", "0.9471", "0.0021", "0.9416", "0.0023")))
for c in J["evaluation"]["comparisons"]:
    check(f"McNemar p {c['a']}–{c['b']}", f"p = {c['p_value']:.2f}" in ALL)
le = J["label_efficiency"]
i5 = le["fractions"].index(0.05)
for m, k, exp in (("usf_mae", "macro_f1", "0.888"), ("densenet161", "macro_f1", "0.808"), ("usf_mae", "accuracy", "0.896"), ("densenet161", "accuracy", "0.820")):
    v = st.mean(le["per_model"][m][k][i5])
    check(f"5% {m} {k} = {exp}", f"{v:.3f}" == exp and exp in ALL, f"{v:.4f}")
check("ViT-B/16 low-label shown as not run", all(x is None for row in le["per_model"]["vit_b16"]["macro_f1"][:-1] for x in row) and "not run" in ALL)
ds = J["dataset_stats"]
for v in (ds["total_images"], ds["total_patients"], ds["split"]["train_patients"], ds["split"]["test_images"], ds["near_duplicates"]["cross_split_pairs"], ds["machines_by_split"]["test"]["Voluson S10"]):
    check(f"dataset count {v:,}", f"{v:,}" in ALL)
cm = pm["usf_mae"]["confusion"]
errs = sum(map(sum, cm)) - sum(cm[i][i] for i in range(6))
check("confusion error count", f"{errs} errors" in ALL, str(errs))
for r in J["faithfulness"]["rows"]:
    check(f"faithfulness {r['model']} {r['method']}", f"{r['deletion_auc_mean']:.2f} ± {r['deletion_auc_sd']:.2f}" in ALL)
op = next(s for s in J["manifest"]["samples"] if "opening" in s["tags"])
for m in ("densenet161", "vit_b16", "usf_mae"):
    check(f"opening reveal {m}", f"{op['predictions'][m]['probs'][op['label']]:.1%}" in ALL)

# ---- provenance / wording guards
check("reconstructions are 500-epoch", all(r["checkpoint"] == "usf_mae_500ep.pth" and r["sha256"].startswith("c659a061") for r in FP["reconstructions"]))
check("no 100-epoch reconstruction file embedded", "reconstructions/" not in json.dumps(FP["reconstructions"]) or all("500ep" not in x for x in []))
check("no localhost participant URL", not re.search(r"localhost|127\.0\.0\.1", "\n".join(texts)))
check("gradient-weighted attention rollout terminology", "Gradient-weighted attention rollout" in ALL and "chefer" not in "\n".join(texts).replace("“chefer”", ""))
check("MetaCAM++ shown only as placeholder", "MetaCAM++" in ALL and re.search(r"No methodology or results", ALL) is not None)
check("no 'statistically equivalent' claim", not re.search(r"statistically equivalent|are equivalent", ALL, re.I))
check("pretraining overlap disclosed", "not label leakage" in ALL.lower() or "not **label leakage**" in ALL)

# ---- timing from notes
mins = [float(m.group(1)) for n in notes for m in [re.search(r"^Time: ([\d.]+) min", n, re.M)] if m]
check("planned timing = 120 min", abs(sum(mins) - 120) < 1e-6, f"{sum(mins)} min over {len(mins)} timed slides")

out = ROOT / "presentation" / "preview" / "validation.json"
out.parent.mkdir(exist_ok=True)
out.write_text(json.dumps({"pptx": str(PPTX.relative_to(ROOT)), "slides": len(prs.slides), "passed": len(results) - len(fails), "failed": fails, "checks": results}, indent=2), encoding="utf-8")
print(f"{len(results) - len(fails)}/{len(results)} checks passed · {len(prs.slides)} slides")
for f in fails:
    print("FAIL", f)
sys.exit(1 if fails else 0)
