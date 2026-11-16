"""Generate every raster figure used by the OMARI deck from the finalized website bundle.

Real figures (dataset images and saved model outputs) go to assets/figures/.
Rendered equations go to assets/equations/. Nothing here re-runs a model: every
pixel value and number comes from website/public/experiments/ (schema 1.1).

    presentation/.venv/Scripts/python presentation/src/make_figures.py
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BUNDLE = ROOT / "website" / "public" / "experiments"
OUT = ROOT / "presentation" / "assets"
FIG = OUT / "figures"
EQ = OUT / "equations"

# Okabe-Ito model colours shared with the website (global.css --m-*)
COL = {"densenet161": "#0072b2", "vit_b16": "#e69f00", "usf_mae": "#009e73"}
INK, INK2, RULE = "#18232b", "#3d4b55", "#d6dde1"
CMAP = "inferno"

OPENING = "Patient01712_Plane2_1_of_2"     # tagged 'opening' in the manifest
CLASS_EXAMPLES = {                          # correctly classified by all three models
    "Fetal abdomen": "Patient01620_Plane2_2_of_2",
    "Fetal brain": "Patient01372_Plane3_1_of_8",
    "Fetal femur": "Patient00647_Plane5_1_of_1",
    "Fetal thorax": "Patient01693_Plane6_1_of_2",
    "Maternal cervix": "Patient01230_Plane4_2_of_4",
    "Other": "Patient01509_Plane1_5_of_13",
}
FEMUR = "Patient00647_Plane5_1_of_1"


def load_json(p: Path):
    return json.loads(p.read_text(encoding="utf-8"))


def gray(path: Path) -> np.ndarray:
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32) / 255.0


def overlay(img: np.ndarray, amap: np.ndarray, alpha_max: float = 0.72) -> np.ndarray:
    """Stretch a low-resolution map over the whole image (models saw a 224x224 stretch of it)."""
    h, w = img.shape
    m = np.asarray(Image.fromarray((amap * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR), dtype=np.float32) / 255.0
    rgb = matplotlib.colormaps[CMAP](m)[..., :3]
    a = (alpha_max * np.clip(m, 0, 1) ** 0.8)[..., None]
    base = np.repeat(img[..., None], 3, axis=2)
    return np.clip(base * (1 - a) + rgb * a, 0, 1)


def save_rgb(arr: np.ndarray, path: Path) -> None:
    Image.fromarray((np.clip(arr, 0, 1) * 255).astype(np.uint8)).save(path, optimize=True)


def sample(man, sid):
    return next(s for s in man["samples"] if s["id"] == sid)


def attribution_png(s, model, method, layer):
    a = next(a for a in s["attributions"] if a["model"] == model and a["method"] == method and a["layer"] == layer)
    return BUNDLE / a["png"], a


def upsample_nearest(path: Path, size: int) -> Image.Image:
    return Image.open(path).convert("L").resize((size, size), Image.NEAREST)


# ---------------------------------------------------------------- figures

def fig_dataset_images(man):
    shutil.copy(BUNDLE / sample(man, OPENING)["image"], FIG / "opening.png")
    for cls, sid in CLASS_EXAMPLES.items():
        shutil.copy(BUNDLE / sample(man, sid)["image"], FIG / f"class_{cls.split()[-1].lower()}.png")


def fig_feature_maps(man):
    s = sample(man, OPENING)
    for f in s["features"]["densenet161"]:
        ch = f["channels"][0]  # highest mean activation, as exported
        img = upsample_nearest(BUNDLE / ch["png"], 448)
        rgb = (matplotlib.colormaps["viridis"](np.asarray(img) / 255.0)[..., :3] * 255).astype(np.uint8)
        Image.fromarray(rgb).save(FIG / f"feat_{f['layer'].split('.')[-1]}.png")
    return [(f["layer"], f["shape"], f["channels"][0]["index"]) for f in s["features"]["densenet161"]]


def fig_vit_patches(man):
    s = sample(man, OPENING)
    im = Image.open(BUNDLE / s["image"]).convert("L").resize((224, 224), Image.BICUBIC).resize((672, 672), Image.NEAREST).convert("RGB")
    d = ImageDraw.Draw(im)
    for k in range(1, 14):
        d.line([(k * 48, 0), (k * 48, 672)], fill=(230, 159, 0), width=2)
        d.line([(0, k * 48), (672, k * 48)], fill=(230, 159, 0), width=2)
    im.save(FIG / "vit_patch_grid.png")
    # a strip of 8 individual patches (row 7) for the token illustration
    base = Image.open(BUNDLE / s["image"]).convert("L").resize((224, 224), Image.BICUBIC)
    for j, col in enumerate(range(3, 11)):
        base.crop((col * 16, 6 * 16, col * 16 + 16, 7 * 16)).resize((96, 96), Image.NEAREST).save(FIG / f"vit_patch_{j}.png")


def fig_attention(man):
    s = sample(man, OPENING)
    img = gray(BUNDLE / s["image"])
    out = {}
    for model in ("vit_b16", "usf_mae"):
        a = load_json(BUNDLE / s["attention"][model]["cls"])
        c = np.asarray(a["cls"], dtype=np.float32)  # blocks x heads x 196
        for b in (0, 5, 11):
            m = c[b].mean(0).reshape(14, 14)
            m = (m - m.min()) / (m.max() - m.min() + 1e-12)
            save_rgb(overlay(img, m), FIG / f"attn_{model}_block{b + 1}.png")
            top = np.argsort(c[b].mean(0))[-5:][::-1]
            out[f"{model}_block{b + 1}"] = {"top5_patches": top.tolist(), "max_share": float(c[b].mean(0).max())}
    return out


def fig_reconstructions(man):
    s = sample(man, OPENING)
    base = np.asarray(Image.open(BUNDLE / s["image"]).convert("L").resize((224, 224), Image.NEAREST), dtype=np.float32) / 255.0
    rows = []
    for r in s["reconstructions"]:
        prov = r["provenance"]
        assert prov["checkpoint_id"] == "usf_mae_500ep.pth", prov["checkpoint_id"]
        assert prov["checkpoint_sha256"].startswith("c659a06170b0"), prov["checkpoint_sha256"]
        pct = int(round(r["mask_ratio"] * 100))
        masked = np.repeat(base[..., None], 3, axis=2).copy()
        for p in r["masked"]:
            i, j = divmod(p, 14)
            masked[i * 16:(i + 1) * 16, j * 16:(j + 1) * 16] = np.array([0.82, 0.86, 0.88])
        Image.fromarray((masked * 255).astype(np.uint8)).resize((448, 448), Image.NEAREST).save(FIG / f"mae_masked_{pct}.png")
        Image.open(BUNDLE / r["pasted_png"]).convert("L").resize((448, 448), Image.NEAREST).save(FIG / f"mae_recon_{pct}.png")
        rows.append({"mask_ratio": r["mask_ratio"], "n_masked": len(r["masked"]), "loss_masked_mse": r["loss_masked_mse"],
                     "checkpoint": prov["checkpoint_id"], "sha256": prov["checkpoint_sha256"]})
    Image.fromarray((base * 255).astype(np.uint8)).resize((448, 448), Image.NEAREST).save(FIG / "mae_input.png")
    return rows


XAI_SET = [  # (model, method, layer, file stem, label)
    ("densenet161", "gradcam", "act", "xai_densenet_gradcam", "Grad-CAM · final features (7×7)"),
    ("densenet161", "gradcampp", "act", "xai_densenet_gradcampp", "Grad-CAM++ · final features (7×7)"),
    ("densenet161", "cam", "act", "xai_densenet_cam", "CAM · final features (7×7)"),
    ("vit_b16", "grad_rollout", "final", "xai_vit_gradrollout", "Gradient-weighted attention rollout"),
    ("vit_b16", "vit_gradcam", "block11", "xai_vit_tokengradcam_b11", "Token Grad-CAM · after block 11 of 12"),
    ("usf_mae", "grad_rollout", "final", "xai_usf_gradrollout", "Gradient-weighted attention rollout"),
    ("usf_mae", "attention_rollout", "final", "xai_usf_rollout", "Attention rollout"),
    ("usf_mae", "vit_gradcam", "block11", "xai_usf_tokengradcam_b11", "Token Grad-CAM · after block 11 of 12"),
    ("usf_mae", "vit_gradcampp", "block11", "xai_usf_tokengradcampp_b11", "Token Grad-CAM++ · after block 11 of 12"),
]


def fig_xai(man):
    s = sample(man, OPENING)
    img = gray(BUNDLE / s["image"])
    rec = []
    for model, method, layer, stem, label in XAI_SET:
        p, a = attribution_png(s, model, method, layer)
        m = gray(p)
        save_rgb(overlay(img, m), FIG / f"{stem}.png")
        rec.append({"file": stem, "model": model, "method": method, "layer": layer, "label": label,
                    "target_class": a["target_class"], "source_png": a["png"],
                    "checkpoint_sha256": a["provenance"]["checkpoint_sha256"]})
    # colour bar
    fig, ax = plt.subplots(figsize=(4, 0.32), dpi=300)
    ax.imshow(np.linspace(0, 1, 256)[None, :], aspect="auto", cmap=CMAP)
    ax.set_axis_off()
    fig.savefig(FIG / "colorbar.png", bbox_inches="tight", pad_inches=0, transparent=True)
    plt.close(fig)
    return rec


def fig_perturbations(man):
    res = {}
    for tag, sid in (("femur", FEMUR), ("abdomen", OPENING)):
        s = sample(man, sid)
        p = next(p for p in s["perturbations"] if p["type"] == "occlude_patch")
        shutil.copy(BUNDLE / s["image"], FIG / f"detective_{tag}_orig.png")
        shutil.copy(BUNDLE / p["image_png"], FIG / f"detective_{tag}_blur.png")
        base = s.get("predictions_display_png") or s["predictions"]
        res[tag] = {"id": sid, "label": s["label"], "perturbation": p["label"],
                    "before": {m: {"pred": v["pred"], "p_true": v["probs"][s["label"]]} for m, v in base.items()},
                    "after": {m: {"pred": v["pred"], "p_true": v["probs"][s["label"]]} for m, v in p["predictions"].items()},
                    "baseline_source": "predictions_display_png" if s.get("predictions_display_png") else "predictions"}
    return res


def fig_label_efficiency():
    le = load_json(BUNDLE / "label_efficiency.json")
    fr = np.array(le["fractions"]) * 100
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 5.0), dpi=240, sharey=True)
    summary = {}
    for ax, metric, title in zip(axes, ("macro_f1", "accuracy"), ("Macro-F1", "Accuracy")):
        for model, name in (("densenet161", "DenseNet-161"), ("usf_mae", "USF-MAE (500-ep)")):
            grid = le["per_model"][model][metric]
            xs, mu, sd = [], [], []
            for f, vals in zip(fr, grid):
                v = [x for x in vals if x is not None]
                if not v:
                    continue
                ax.scatter([f] * len(v), v, s=40, color=COL[model], alpha=0.5, linewidths=0, zorder=3)
                xs.append(f); mu.append(np.mean(v)); sd.append(np.std(v, ddof=1) if len(v) > 1 else np.nan)
            ax.plot(xs, mu, color=COL[model], lw=3.2, marker="o", ms=8, label=name, zorder=4)
            summary[(model, metric)] = list(zip(xs, mu, sd))
        vit = le["per_model"]["vit_b16"][metric][-1][0]
        ax.scatter([100], [vit], marker="D", s=58, color=COL["vit_b16"], edgecolor="white", linewidth=0.8, zorder=5, label="ViT-B/16 (100% only)")
        ax.axvspan(3.6, 7.0, color="#e9eef1", zorder=0)
        ax.set_xscale("log")
        ax.set_xticks(fr)
        ax.set_xticklabels([f"{f:g}%" for f in fr])
        ax.minorticks_off()
        ax.set_xlim(0.8, 125)
        ax.set_ylim(0.45, 1.0)
        ax.set_title(title, loc="left", fontsize=19, color=INK, fontweight="semibold", pad=8)
        ax.set_xlabel("Labelled training patients (log scale)", fontsize=15, color=INK2)
        ax.grid(axis="y", color=RULE, lw=0.8)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color("#9aa7ae")
        ax.tick_params(colors=INK2, labelsize=14.5)
    axes[0].set_ylabel("Test-set score (n = 5,271)", fontsize=15, color=INK2)
    axes[1].legend(loc="lower right", frameon=False, fontsize=14)
    fig.text(0.01, -0.02, "Dots = individual seeds (0, 1, 2); line = seed mean. DenseNet-161 at 100% is seed 0 only. "
             "ViT-B/16 below 100% was not run.", fontsize=13, color=INK2)
    fig.tight_layout()
    fig.savefig(FIG / "label_efficiency.png", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return {f"{m}/{k}": [(float(a), float(b), (None if np.isnan(c) else float(c))) for a, b, c in v] for (m, k), v in summary.items()}


EQUATIONS = {
    "eq_conv": r"$y_{i,j}=\sum_{u,v}\,w_{u,v}\;x_{i+u,\,j+v}+b$",
    "eq_dense": r"$x_\ell = H_\ell\left([\,x_0,\,x_1,\,\ldots,\,x_{\ell-1}\,]\right)$",
    "eq_attention": r"$\mathrm{Attention}(Q,K,V)=\mathrm{softmax}\!\left(\frac{QK^{\top}}{\sqrt{d_k}}\right)V$",
    "eq_mae": r"$\mathcal{L}_{\mathrm{MAE}}=\frac{1}{|M|}\sum_{p\in M}\left\Vert\,\hat{x}_p-x_p\,\right\Vert_2^2$",
    "eq_macro_f1": r"$\mathrm{macro\,F_1}=\frac{1}{K}\sum_{c=1}^{K}\frac{2\,\mathrm{P}_c\,\mathrm{R}_c}{\mathrm{P}_c+\mathrm{R}_c}$",
    "eq_temperature": r"$\hat{p}_i=\mathrm{softmax}(z_i\,/\,T)$",
    "eq_gradcam": r"$\alpha_k^{c}=\frac{1}{Z}\sum_{i,j}\frac{\partial y^{c}}{\partial A^{k}_{ij}}\qquad L^{c}=\mathrm{ReLU}\!\left(\sum_k \alpha_k^{c}A^{k}\right)$",
}


def render_equations():
    plt.rcParams["mathtext.fontset"] = "cm"
    for name, tex in EQUATIONS.items():
        fig = plt.figure(figsize=(0.01, 0.01), dpi=400)
        fig.text(0, 0, tex, fontsize=26, color=INK)
        fig.savefig(EQ / f"{name}.png", bbox_inches="tight", pad_inches=0.04, transparent=True)
        plt.close(fig)


def main():
    for d in (FIG, EQ):
        d.mkdir(parents=True, exist_ok=True)
    man = load_json(BUNDLE / "manifest.json")
    assert man["schema_version"] == "1.1", man["schema_version"]
    fig_dataset_images(man)
    log = {
        "bundle": str(BUNDLE.relative_to(ROOT)),
        "bundle_finalized_at": man["finalizer"]["finalized_at"],
        "opening_sample": OPENING,
        "class_examples": CLASS_EXAMPLES,
        "feature_maps": fig_feature_maps(man),
        "attention": fig_attention(man),
        "reconstructions": fig_reconstructions(man),
        "xai": fig_xai(man),
        "perturbations": fig_perturbations(man),
        "label_efficiency": fig_label_efficiency(),
    }
    fig_vit_patches(man)
    render_equations()
    (OUT / "figure_provenance.json").write_text(json.dumps(log, indent=2), encoding="utf-8")
    print("figures:", len(list(FIG.glob("*.png"))), "equations:", len(list(EQ.glob("*.png"))))


if __name__ == "__main__":
    main()
