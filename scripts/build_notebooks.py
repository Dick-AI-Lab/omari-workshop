#!/usr/bin/env python
"""Generate the six standalone Colab notebooks in notebooks/ from this single source.

    python scripts/build_notebooks.py

Every notebook starts with the same setup cell (repository + packages + data + artifact folders) and has three modes:
  smoke  synthetic images + random weights: checks the code path end to end (numbers are meaningless)
  demo   real FETAL_PLANES_DB subset + real pretrained weights, few epochs (≈10–20 min on a Colab T4)
  full   complete experiments (hours on a single GPU)
"""
import os
import textwrap

import nbformat as nbf

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "notebooks")

SETUP = r'''#@title 0 · Setup: repository, packages, data and artifact folders { display-mode: "form" }
import os, sys, subprocess, json, random, time
MODE = "demo"  #@param ["smoke", "demo", "full"]
REPO_URL = "https://github.com/YOUR-ORG/omari-workshop.git"  #@param {type:"string"}
DATA_ROOT = "/content/data/FETAL_PLANES_ZENODO"  #@param {type:"string"}
ARTIFACTS = "/content/omari_artifacts"  #@param {type:"string"}
USE_DRIVE_FOR_ARTIFACTS = False  #@param {type:"boolean"}
MODE = os.environ.get("OMARI_MODE", MODE)            # lets CI / nbclient override
ARTIFACTS = os.environ.get("OMARI_ARTIFACTS", ARTIFACTS)
IN_COLAB = "google.colab" in sys.modules

if IN_COLAB and USE_DRIVE_FOR_ARTIFACTS:
    from google.colab import drive
    drive.mount("/content/drive")
    ARTIFACTS = "/content/drive/MyDrive/omari_artifacts"

# 1) make `omari_us` importable: installed package > local checkout (../python) > clone REPO_URL
def _find_repo():
    for cand in [os.getcwd(), os.path.dirname(os.getcwd()), "/content/omari-workshop"]:
        if os.path.isdir(os.path.join(cand, "python", "omari_us")):
            return cand
    return None
REPO = _find_repo()
if REPO is None:
    if "YOUR-ORG" in REPO_URL:
        raise RuntimeError("Set REPO_URL to the repository that contains python/omari_us (or upload the repo to /content/omari-workshop).")
    subprocess.run(["git", "clone", "--depth", "1", REPO_URL, "/content/omari-workshop"], check=True)
    REPO = "/content/omari-workshop"
sys.path.insert(0, os.path.join(REPO, "python"))

# 2) packages (Colab already has torch/torchvision; timm and friends are added)
if IN_COLAB:
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "timm>=0.9.12", "scikit-learn", "scipy", "pyyaml", "opencv-python-headless", "gdown"], check=True)

import numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
import omari_us
from omari_us.utils import seed_everything, get_device, CLASSES, CLASS_SHORT
SEED = 42
seed_everything(SEED)
DEVICE = get_device()
os.makedirs(ARTIFACTS, exist_ok=True)
SPLITS_CSV = os.path.join(ARTIFACTS, "splits", "split_manifest.csv")
RUNS = os.path.join(ARTIFACTS, "runs")
print(f"omari_us {omari_us.__version__} | torch {torch.__version__} | device {DEVICE} | MODE={MODE} | artifacts → {ARTIFACTS}")
if DEVICE.type == "cpu" and MODE != "smoke":
    print("⚠ No GPU. In Colab: Runtime → Change runtime type → T4 GPU. CPU training of these models is impractically slow.")
'''

DATA = r'''#@title 1 · Data: download FETAL_PLANES_DB (2.1 GB, CC BY 4.0) or build the synthetic smoke dataset { display-mode: "form" }
from omari_us.datasets import read_metadata, make_splits, synthetic_metadata, save_split_manifest, load_split_manifest, ZENODO_URL, ZENODO_MD5

def ensure_fetal_planes(root=DATA_ROOT):
    """Download + verify + extract the Zenodo archive if the metadata CSV is not already present."""
    import glob, hashlib
    if glob.glob(os.path.join(root, "**", "FETAL_PLANES_DB_data.csv"), recursive=True):
        return
    os.makedirs(root, exist_ok=True)
    z = os.path.join(os.path.dirname(root), "FETAL_PLANES_ZENODO.zip")
    if not os.path.exists(z):
        print("downloading", ZENODO_URL)
        subprocess.run(["curl", "-L", "--fail", "-o", z, ZENODO_URL], check=True)
    md5 = hashlib.md5(open(z, "rb").read()).hexdigest()
    assert md5 == ZENODO_MD5, f"md5 mismatch: {md5}"
    subprocess.run(["unzip", "-q", "-n", z, "-d", root], check=True)

if MODE == "smoke":
    SMOKE_DIR = os.path.join(ARTIFACTS, "smoke_data")
    meta = synthetic_metadata(60, (2, 5), seed=0, image_dir=os.path.join(SMOKE_DIR, "Images"), size=(200, 150))
    print("SMOKE MODE: synthetic grey images; every number below is meaningless.")
else:
    ensure_fetal_planes()
    meta = read_metadata(DATA_ROOT)
print(meta.shape)
meta.head()
'''


def md(s):
    return nbf.v4.new_markdown_cell(textwrap.dedent(s).strip())


def code(s):
    return nbf.v4.new_code_cell(textwrap.dedent(s).strip())


def header(n, title, objective, runtime, inputs, outputs):
    return md(f"""
    # {n:02d} · {title}
    **OMARI Seminar Series — From Pixels to Predictions: Developing AI Models for Medical Image Interpretation**

    {objective}

    | | |
    |---|---|
    | **Runtime** | {runtime} |
    | **Needs** | {inputs} |
    | **Produces** | {outputs} |
    | **Modes** | `smoke` (synthetic data, random weights: checks the code only) · `demo` (real data subset, few epochs) · `full` (complete experiment) |

    > Results from `smoke` mode are meaningless by construction and must never be exported to the website.
    > Every artifact written by these notebooks records the script, checkpoint hash and parameters that produced it.
    """)


REFS = md("""
## References
- Burgos-Artizzu X.P. et al. *Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes.* Sci. Rep. 10, 10200 (2020). Dataset: Zenodo 3904280, CC BY 4.0.
- Huang G. et al. *Densely connected convolutional networks.* CVPR 2017.
- Dosovitskiy A. et al. *An image is worth 16×16 words.* ICLR 2021; Steiner A. et al. *How to train your ViT?* TMLR 2022.
- He K. et al. *Masked autoencoders are scalable vision learners.* CVPR 2022.
- Megahed Y. et al. *USF-MAE: Ultrasound self-supervised foundation model with masked autoencoding.* Biomed. Signal Process. Control 122, 110313 (2026). https://github.com/Yusufii9/USF-MAE
- Zhou B. et al. CAM, CVPR 2016 · Selvaraju R.R. et al. Grad-CAM, ICCV 2017 · Chattopadhay A. et al. Grad-CAM++, WACV 2018 · Abnar & Zuidema, attention rollout, ACL 2020 · Chefer H. et al., CVPR 2021 · Adebayo J. et al., sanity checks, NeurIPS 2018 · Petsiuk V. et al., RISE / deletion–insertion, BMVC 2018 · Guo C. et al., calibration, ICML 2017.
""")

LOAD_SPLITS = r'''# Load the split manifest written by notebook 01 (or rebuild it identically from the metadata)
if os.path.exists(SPLITS_CSV) and MODE != "smoke":
    splits = load_split_manifest(SPLITS_CSV, DATA_ROOT)
else:
    splits = make_splits(meta, val_fraction=0.15 if MODE == "smoke" else 0.1, seed=SEED)
    save_split_manifest(splits, SPLITS_CSV)
if MODE == "demo":  # stratified patient subset for speed
    keep = splits.groupby("split", group_keys=False).apply(lambda d: d[d.Patient_num.isin(pd.Series(d.Patient_num.unique()).sample(frac=0.25, random_state=SEED))])
    splits = keep.reset_index(drop=True)
print(splits.groupby(["split"]).agg(images=("Image_name", "size"), patients=("Patient_num", "nunique")))
'''


def train_cells(model_id, cfg_name, extra_kw=""):
    return [
        md(f"""
        ## Train: frozen encoder (linear probe) and full fine-tuning
        Both regimes use the same patient-disjoint splits, class-weighted cross-entropy with label smoothing, cosine schedule with warm-up,
        and model selection by **validation macro-F1**. The test split is evaluated once with the selected checkpoint.
        Hyper-parameters come from `configs/{cfg_name}`; `demo` shortens epochs, `smoke` uses random weights and 2 batches.
        """),
        code(f'''
        from omari_us.training import TrainConfig, train
        from omari_us.utils import load_yaml
        C = load_yaml(os.path.join(REPO, "configs", "{cfg_name}"))
        def make_cfg(regime, label_fraction=1.0, seed=0):
            r = dict(C["regimes"][regime])
            cfg = TrainConfig(model="{model_id}", regime=regime, label_fraction=label_fraction, seed=seed, out_dir=RUNS,
                              loss=C["loss"], weight_scheme=C["weight_scheme"], label_smoothing=C["label_smoothing"], patience=C["patience"],
                              input_overrides=C.get("input", {{}}), augment=C.get("augment", {{}}){extra_kw}, **r)
            if MODE == "demo":
                cfg.epochs = min(cfg.epochs, 4)
            if MODE == "smoke":
                cfg.epochs, cfg.batch_size, cfg.max_train_batches, cfg.max_eval_batches = 2, 4, 2, 2
                cfg.pretrained, cfg.num_workers, cfg.amp = False, 0, False
            return cfg

        results = {{}}
        for regime in ["frozen", "finetune"]:
            t0 = time.time()
            results[regime] = train(make_cfg(regime), splits)
            s = results[regime]
            print(f"{{regime:9s}} best epoch {{s['best_epoch']}} | val macro-F1 {{s['best_val_macro_f1']:.3f}} | test macro-F1 {{s['test']['macro_f1']:.3f}} | {{time.time()-t0:.0f}}s")
        '''),
        code('''
        # Learning curves (written to history.json in each run folder)
        fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
        for ax, regime in zip(axes, ["frozen", "finetune"]):
            h = json.load(open(os.path.join(RUNS, results[regime]["run"], "history.json")))
            ax.plot(h["epoch"], h["train_loss"], label="train loss"); ax.plot(h["epoch"], h["val_loss"], label="val loss")
            ax2 = ax.twinx(); ax2.plot(h["epoch"], h["val_macro_f1"], "g--", label="val macro-F1"); ax2.set_ylim(0, 1)
            ax.set_title(regime); ax.set_xlabel("epoch"); ax.legend(loc="upper left"); ax2.legend(loc="upper right")
        plt.tight_layout(); plt.show()
        '''),
        code('''
        from omari_us.evaluation import classification_report
        z = np.load(os.path.join(RUNS, results["finetune"]["run"], "test_predictions.npz"), allow_pickle=True)
        rep = classification_report(z["labels"], z["probs"].argmax(1))
        print(f"test accuracy {rep['accuracy']:.3f} | macro-F1 {rep['macro_f1']:.3f} | balanced accuracy {rep['balanced_accuracy']:.3f}")
        pd.DataFrame(rep["per_class"], index=CLASS_SHORT).round(3)
        '''),
        md("""
        ### Optional: label-efficiency sweep
        Repeats training on patient-level label fractions (1–100%) × seeds. Expensive in `full` mode (18 runs per regime);
        notebook 05 collects the `summary.json` files into the website's learning-curve figure.
        """),
        code('''
        RUN_LABEL_SWEEP = False  #@param {type:"boolean"}
        SWEEP_REGIME = "frozen"  #@param ["frozen", "finetune"]
        if RUN_LABEL_SWEEP:
            fracs = C["label_fractions"] if MODE == "full" else [0.05, 0.25, 1.0]
            seeds = C["seeds"] if MODE == "full" else [0]
            for f in fracs:
                for s in seeds:
                    r = train(make_cfg(SWEEP_REGIME, f, s), splits, progress=False)
                    print(r["run"], round(r["test"]["macro_f1"], 4))
        '''),
    ]


def nb01():
    c = [header(1, "Ultrasound dataset preparation", "Load FETAL_PLANES_DB, examine class balance and acquisition metadata, build **patient-level** splits that keep the authors' test set, detect near-duplicate images, visualise each model's preprocessing, and save reusable split manifests plus the statistics used by the website.",
                "CPU is fine (≈5 min; duplicate search ≈5–10 min for 12,400 images)", "Internet access to zenodo.org (or the extracted archive)", "`splits/split_manifest.csv`, `splits/near_duplicates.csv`, `dataset_stats.json`"),
         code(SETUP), code(DATA),
         md("""
         ## 1 · Metadata and class balance
         The CSV has one row per image: `Image_name; Patient_num; Plane; Brain_plane; Operator; US_Machine; Train`. `Train` holds the authors' split
         (patients ordered by first visit; first half → train). Column names carry trailing spaces in the release, which `read_metadata` strips.
         """),
         code('''
         counts = pd.crosstab(meta["Plane"], meta["Train"].map({1: "train", 0: "test"}))
         counts["total"] = counts.sum(1)
         display(counts.sort_values("total", ascending=False))
         ax = counts["total"].sort_values().plot.barh(figsize=(6, 3), color="#3d4b55"); ax.set_xlabel("images"); plt.show()
         print(f"{meta.Patient_num.nunique()} patients, {len(meta)} images, {len(meta)/meta.Patient_num.nunique():.2f} images/patient")
         '''),
         md("## 2 · Representative images (two per class)"),
         code('''
         from PIL import Image
         fig, axes = plt.subplots(2, 6, figsize=(15, 5))
         for c in range(6):
             rows = meta[meta.label == c].sample(2, random_state=SEED)
             for r, (_, row) in enumerate(rows.iterrows()):
                 im = Image.open(row.path).convert("L"); axes[r, c].imshow(im, cmap="gray"); axes[r, c].axis("off")
                 axes[r, c].set_title(f"{CLASS_SHORT[c]}\\n{im.size[0]}×{im.size[1]}", fontsize=9)
         plt.tight_layout(); plt.show()
         '''),
         md("## 3 · Acquisition metadata: machines, operators, image sizes\nAcquisition variables that correlate with class are candidate shortcuts (Chapter 6) and sources of domain shift (Chapter 4)."),
         code('''
         if "US_Machine" in meta: display(pd.crosstab(meta["Plane"], meta["US_Machine"], normalize="columns").round(2))
         if "Operator" in meta: display(pd.crosstab(meta["Plane"], meta["Operator"]).T)
         sample = meta.sample(min(800, len(meta)), random_state=SEED)
         sizes = pd.Series([Image.open(p).size for p in sample.path], index=sample.index)
         print("most common image sizes (sample):"); print(sizes.value_counts().head(8))
         '''),
         md("""
         ## 4 · Patient-level splits
         We keep the authors' test split and carve a **patient-disjoint, class-stratified** validation set (10%) from the training half
         with `StratifiedGroupKFold`. `check_patient_disjoint` raises if any patient appears in two splits.
         """),
         code('''
         splits = make_splits(meta, val_fraction=0.15 if MODE == "smoke" else 0.1, seed=SEED)
         from omari_us.datasets import check_patient_disjoint
         print("patients per split:", check_patient_disjoint(splits))
         display(pd.crosstab(splits["Plane"], splits["split"]))
         '''),
         md("### What a random image-level split would have done\nThe share of test images whose patient also has images in training, measured empirically."),
         code('''
         from sklearn.model_selection import train_test_split
         tr, te = train_test_split(meta, test_size=0.4, random_state=SEED, stratify=meta.label)
         leak = te.Patient_num.isin(set(tr.Patient_num)).mean()
         print(f"random image-level split: {leak:.1%} of test images share a patient with training (patient-level split: 0%)")
         '''),
         md("## 5 · Near-duplicate detection\n64-bit difference hashes; pairs within `MAX_HAMMING` bits are flagged. Inspect a few pairs before drawing conclusions: dHash also matches different frames of very similar views."),
         code('''
         from omari_us.datasets import find_near_duplicates
         MAX_HAMMING = 4  #@param {type:"integer"}
         dup_df = splits if MODE == "full" else splits.sample(min(3000, len(splits)), random_state=SEED).reset_index(drop=True)
         dups = find_near_duplicates(dup_df, MAX_HAMMING)
         print(f"{len(dups)} near-duplicate pairs in {len(dup_df)} images; same patient: {dups.same_patient.mean() if len(dups) else 0:.1%}")
         if len(dups):
             cross = dups[(dups.split_a == "test") ^ (dups.split_b == "test")]
             print(f"pairs crossing the train/val ↔ test boundary: {len(cross)}")
             fig, axes = plt.subplots(2, min(5, len(dups)), figsize=(14, 5), squeeze=False)
             for k, (_, r) in enumerate(dups.head(axes.shape[1]).iterrows()):
                 for j, idx in enumerate((r.a, r.b)):
                     axes[j, k].imshow(Image.open(dup_df.path.iloc[idx]).convert("L"), cmap="gray"); axes[j, k].axis("off")
                     axes[j, k].set_title(dup_df.Image_name.iloc[idx][:22], fontsize=7)
             plt.show()
         '''),
         md("## 6 · Preprocessing and augmentation, per model\nEach model keeps the preprocessing its pretrained weights expect (see `omari_us/datasets/transforms.py`)."),
         code('''
         from omari_us.datasets import input_spec, build_transforms, denormalize
         row = splits[splits.split == "train"].iloc[0]
         im = Image.open(row.path).convert("L")
         fig, axes = plt.subplots(3, 5, figsize=(14, 8))
         for i, mid in enumerate(["densenet161", "vit_b16", "usf_mae"]):
             spec = input_spec(mid); ev, tr = build_transforms(spec, False), build_transforms(spec, True)
             axes[i, 0].imshow(denormalize(ev(im), spec), cmap="gray"); axes[i, 0].set_title(f"{mid} eval\\n{spec.interpolation}, μ={spec.mean[0]}", fontsize=8)
             for j in range(1, 5):
                 axes[i, j].imshow(denormalize(tr(im), spec), cmap="gray"); axes[i, j].set_title("train aug", fontsize=8)
             for a in axes[i]: a.axis("off")
         plt.tight_layout(); plt.show()
         '''),
         md("## 7 · Save manifests and website statistics"),
         code('''
         from omari_us.datasets import dataset_stats
         from omari_us.utils import write_json
         save_split_manifest(splits, SPLITS_CSV)
         stats = dataset_stats(splits, dups if MODE == "full" else None, MAX_HAMMING, script="notebooks/01_dataset_preparation.ipynb")
         write_json(stats, os.path.join(ARTIFACTS, "dataset_stats.json"), indent=1)
         if len(dups): dups.to_csv(os.path.join(ARTIFACTS, "splits", "near_duplicates.csv"), index=False)
         print("wrote", SPLITS_CSV, "and dataset_stats.json")
         print("Copy dataset_stats.json to website/public/experiments/ (not in smoke mode) to replace the literature counts on the website.")
         assert stats["split"]["patient_overlap"] == 0
         '''),
         md("""
         ## Expected outputs (full mode)
         - 12,400 images from 1,792 patients; six classes with "Other" the largest and the abdomen the smallest class.
         - `patient_overlap = 0` between the authors' train and test halves; validation ≈10% of training patients.
         - A non-trivial number of near-duplicate pairs, almost all within the same patient (saved frame sequences).
         """), REFS]
    return c


def nb02():
    return [header(2, "DenseNet-161: supervised CNN baseline", "Load ImageNet-pretrained DenseNet-161, replace the 1000-way head with a 6-way head, run frozen-backbone and full fine-tuning experiments, and export feature maps for the website.",
                   "GPU recommended: demo ≈15 min (T4), full ≈1–2 h per regime", "`splits/split_manifest.csv` from notebook 01 (rebuilt automatically if absent)", "`runs/densenet161_*/{best.pt, history.json, test_predictions.npz, summary.json}`"),
            code(SETUP), code(DATA), code(LOAD_SPLITS),
            md("""
            ## Architecture
            `omari_us.models.DenseNet161Classifier` wraps torchvision's DenseNet-161 with an explicit, non-inplace final ReLU (`model.act`).
            torchvision applies `F.relu(..., inplace=True)` inside `forward`, which interferes with activation and gradient hooks. Keys are unchanged, so
            `DenseNet161_Weights.IMAGENET1K_V1` loads directly.
            """),
            code('''
            from omari_us.models import build_model
            model = build_model("densenet161", pretrained=(MODE != "smoke"))
            n = sum(p.numel() for p in model.parameters())
            print(f"init: {model.init_source} | parameters: {n:,} (expected 26,485,254 with a 6-class head)")
            assert n == 26_485_254
            x = torch.randn(1, 3, 224, 224)
            shapes = {}
            hooks = [getattr(model.features, b).register_forward_hook(lambda m, i, o, b=b: shapes.__setitem__(b, tuple(o.shape))) for b in ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]]
            with torch.no_grad(): model(x)
            for h in hooks: h.remove()
            shapes
            '''),
            *train_cells("densenet161", "densenet161.yaml"),
            md("## Feature maps from the fine-tuned model\nTop channels by mean activation at each dense block, for one test image. The website shows the same view from exported PNGs."),
            code('''
            from omari_us.models import load_finetuned
            from omari_us.datasets import input_spec, build_transforms
            from PIL import Image
            ft, meta_ft = load_finetuned("densenet161", os.path.join(RUNS, results["finetune"]["run"], "best.pt"))
            tf = build_transforms(input_spec("densenet161"), False)
            row = splits[splits.split == "test"].iloc[0]
            x = tf(Image.open(row.path).convert("L")).unsqueeze(0)
            acts = {}
            hooks = [getattr(ft.features, b).register_forward_hook(lambda m, i, o, b=b: acts.__setitem__(b, torch.relu(o)[0])) for b in ["denseblock1", "denseblock2", "denseblock3", "denseblock4"]]
            with torch.no_grad(): ft(x)
            for h in hooks: h.remove()
            fig, axes = plt.subplots(4, 8, figsize=(15, 8))
            for i, (b, a) in enumerate(acts.items()):
                top = a.mean((1, 2)).argsort(descending=True)[:8]
                for j, c in enumerate(top):
                    axes[i, j].imshow(a[c], cmap="inferno"); axes[i, j].axis("off"); axes[i, j].set_title(f"{b[-6:]} #{int(c)}", fontsize=7)
            plt.tight_layout(); plt.show()
            '''),
            md("Next: notebook 05 compares models; `scripts/export_website_artifacts.py` exports feature maps, attributions and predictions with provenance."), REFS]


def nb03():
    return [header(3, "Vanilla ViT-B/16: supervised vision transformer baseline", "Load `vit_base_patch16_224.augreg_in1k` (supervised ImageNet-1k, **not** MAE), visualise patch embeddings and learned positional embeddings, fine-tune under the same conditions as the other models, and extract attention for the website.",
                   "GPU recommended: demo ≈15 min (T4), full ≈1–2 h per regime", "`splits/split_manifest.csv`", "`runs/vit_b16_*`, attention visualisations"),
            code(SETUP), code(DATA), code(LOAD_SPLITS),
            md("""
            ## Patches, embeddings and positions
            The 224×224 input becomes 14×14 = 196 patches of 16×16×3 = 768 values, projected by a 16×16/16 convolution (`patch_embed.proj`).
            Below: the first principal components of the learned projection filters (cf. Dosovitskiy et al., Fig. 7) and the cosine similarity of learned positional embeddings.
            """),
            code('''
            from omari_us.models import build_model
            vit = build_model("vit_b16", pretrained=(MODE != "smoke")).eval()
            print("init:", vit.init_source, "| params:", f"{sum(p.numel() for p in vit.parameters()):,}")
            W = vit.patch_embed.proj.weight.detach().reshape(768, -1)       # (768, 3*16*16)
            U, S, Vt = torch.linalg.svd(W - W.mean(0), full_matrices=False)
            fig, axes = plt.subplots(2, 14, figsize=(15, 2.4))
            for k, ax in enumerate(axes.flat):
                f = Vt[k].reshape(3, 16, 16).mean(0); ax.imshow(f, cmap="gray"); ax.axis("off")
            plt.suptitle("principal components of patch-embedding filters"); plt.show()
            P = vit.pos_embed.detach()[0, 1:]                                 # (196, 768)
            sim = torch.nn.functional.cosine_similarity(P[:, None], P[None], dim=-1)
            fig, axes = plt.subplots(3, 7, figsize=(12, 5.5))
            for ax, q in zip(axes.flat, range(0, 196, 10)):
                ax.imshow(sim[q].reshape(14, 14), cmap="viridis"); ax.set_title(f"patch {q}", fontsize=7); ax.axis("off")
            plt.suptitle("learned positional-embedding similarity (random in smoke mode)"); plt.show()
            '''),
            *train_cells("vit_b16", "vit_b16.yaml"),
            md("""
            ## Attention from the fine-tuned model
            timm uses fused scaled-dot-product attention; `omari_us.explainability.vit.capture_attention` turns it off temporarily so the attention
            matrices exist, records them with forward hooks, and restores the setting. Remember: attention is routing, not attribution.
            """),
            code('''
            from omari_us.models import load_finetuned
            from omari_us.datasets import input_spec, build_transforms
            from omari_us.explainability import vit as vitx
            from PIL import Image
            ft, _ = load_finetuned("vit_b16", os.path.join(RUNS, results["finetune"]["run"], "best.pt"))
            row = splits[splits.split == "test"].iloc[0]
            im = Image.open(row.path).convert("L")
            x = build_transforms(input_spec("vit_b16"), False)(im).unsqueeze(0)
            A = vitx.attention_maps(ft, x)                 # (12 blocks, 12 heads, 197, 197)
            print(A.shape, "rows sum to 1:", torch.allclose(A.sum(-1), torch.ones_like(A.sum(-1)), atol=1e-4))
            fig, axes = plt.subplots(2, 7, figsize=(15, 4.6))
            axes[0, 0].imshow(im.resize((224, 224)), cmap="gray"); axes[0, 0].set_title("input"); axes[0, 0].axis("off")
            for b, ax in zip(range(12), list(axes.flat)[1:13]):
                ax.imshow(A[b, :, 0, 1:].mean(0).reshape(14, 14), cmap="inferno"); ax.set_title(f"block {b+1} [CLS]", fontsize=8); ax.axis("off")
            ax = list(axes.flat)[13]; ax.imshow(vitx.attention_rollout(ft, x), cmap="inferno"); ax.set_title("rollout", fontsize=8); ax.axis("off")
            plt.tight_layout(); plt.show()
            '''), REFS]


def nb04():
    return [header(4, "USF-MAE: ultrasound foundation model adaptation", "Retrieve and verify the official 100-epoch USF-MAE checkpoint, transfer its ViT-B/16 encoder into a classifier, demonstrate masked reconstruction (if decoder weights are present), fine-tune on FETAL_PLANES_DB, and export embeddings. USF-MAE is **not** pretrained here.",
                   "GPU recommended: demo ≈15 min, full ≈1–2 h per regime", "Google Drive access for the checkpoint (gdown); `splits/split_manifest.csv`", "`checkpoints/usf_mae_100ep.pth`, `runs/usf_mae_*`, `usf_mae_test_embeddings.npz`"),
            code(SETUP), code(DATA), code(LOAD_SPLITS),
            md("""
            ## 1 · Retrieve the checkpoint
            Source: https://github.com/Yusufii9/USF-MAE (README → Checkpoints → *USF-MAE (100 epochs – recommended)*; Google Drive id `1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr`).
            In `smoke` mode a randomly initialised MAE checkpoint in the same format is created instead.
            """),
            code('''
            CKPT = os.path.join(ARTIFACTS, "checkpoints", "usf_mae_100ep.pth")
            os.makedirs(os.path.dirname(CKPT), exist_ok=True)
            from omari_us.models.usf_mae import USF_MAE_100EP_GDRIVE_ID, mae_vit_base_patch16_dec512d8b
            if MODE == "smoke":
                torch.save({"model": mae_vit_base_patch16_dec512d8b().state_dict()}, CKPT)
            elif not os.path.exists(CKPT):
                import gdown
                gdown.download(id=USF_MAE_100EP_GDRIVE_ID, output=CKPT, quiet=False)
            print(CKPT, f"{os.path.getsize(CKPT)/1e6:.0f} MB")
            '''),
            md("## 2 · Verify compatibility\n`load_usf_mae_checkpoint` strips `module.`/`model.` prefixes, compares every tensor with `mae_vit_base_patch16_dec512d8b`, checks the positional embedding equals MAE's fixed sin-cos table, and reports whether decoder weights exist."),
            code('''
            from omari_us.models.usf_mae import load_usf_mae_checkpoint
            sd, report = load_usf_mae_checkpoint(CKPT)
            print(report.summary())
            assert report.ok_for_classification, "encoder unusable"
            '''),
            md("""
            ## 3 · Masked reconstruction (requires decoder weights)
            Identical preprocessing to the official loader: `Resize((224,224), NEAREST)`, ImageNet mean/std. The loss is MSE on masked patches of normalised
            pixels (`norm_pix_loss=False`). Images shown are de-normalised and averaged over channels.
            """),
            code('''
            from omari_us.models.usf_mae import build_mae_from_state_dict, reconstruct
            from omari_us.datasets import input_spec, build_transforms, denormalize
            from PIL import Image
            spec = input_spec("usf_mae"); tf = build_transforms(spec, False)
            if report.ok_for_reconstruction:
                mae = build_mae_from_state_dict(sd).to(DEVICE)
                row = splits[splits.split == "test"].iloc[0]
                x = tf(Image.open(row.path).convert("L")).unsqueeze(0).to(DEVICE)
                ratios = [0.25, 0.5, 0.75, 0.9]
                fig, axes = plt.subplots(2, len(ratios) + 1, figsize=(15, 6))
                axes[0, 0].imshow(denormalize(x[0], spec).cpu(), cmap="gray"); axes[0, 0].set_title("input"); axes[1, 0].axis("off")
                for k, r in enumerate(ratios, 1):
                    out = reconstruct(mae, x, r, seed=0)
                    m = out["mask"][0].reshape(14, 14).repeat_interleave(16, 0).repeat_interleave(16, 1).cpu()
                    axes[0, k].imshow(denormalize(x[0], spec).cpu() * (1 - m), cmap="gray"); axes[0, k].set_title(f"{int(r*100)}% masked")
                    axes[1, k].imshow(denormalize(out["pasted"][0], spec).cpu(), cmap="gray"); axes[1, k].set_title(f"reconstruction (MSE {out['loss']:.3f})")
                for a in axes.flat: a.axis("off")
                plt.tight_layout(); plt.show()
            else:
                print("Decoder weights absent: no reconstruction is shown (nothing is synthesised in its place).")
            '''),
            md("""
            ## 4 · Dataset overlap (must be reported)
            FETAL_PLANES_DB is dataset 25 of OpenUS-46, the USF-MAE pretraining corpus. Its images, **without labels**, were therefore seen during
            self-supervised pretraining, possibly including our test images. This is a transductive advantage over the ImageNet-initialised baselines.
            Report it with every USF-MAE result on this dataset. A clean test needs a USF-MAE variant pretrained without FETAL_PLANES_DB, or an external fetal test set.
            The public pretraining notebook also initialises from Facebook's ImageNet MAE checkpoint (`mae_pretrain_vit_base_full.pth`), so USF-MAE = ImageNet MAE → ultrasound MAE.
            """),
            *train_cells("usf_mae", "usf_mae.yaml", extra_kw=', usf_mae_checkpoint=CKPT, usf_mae_global_pool=C.get("global_pool", "token")'),
            md("## 5 · Embeddings for the test set\n[CLS] features of the fine-tuned encoder, saved for notebook 05 and visualised with PCA."),
            code('''
            from omari_us.models import load_finetuned
            from omari_us.datasets import FetalPlanesDataset
            from torch.utils.data import DataLoader
            ft, _ = load_finetuned("usf_mae", os.path.join(RUNS, results["finetune"]["run"], "best.pt"))
            ft = ft.to(DEVICE).eval()
            te = splits[splits.split == "test"]
            if MODE != "full": te = te.sample(min(600, len(te)), random_state=SEED)
            dl = DataLoader(FetalPlanesDataset(te, tf), batch_size=64)
            feats, labs = [], []
            with torch.no_grad():
                for x, y, _ in dl:
                    feats.append(ft.forward_head(ft.forward_features(x.to(DEVICE)), pre_logits=True).cpu()); labs.append(y)
            F_, Y = torch.cat(feats).numpy(), torch.cat(labs).numpy()
            np.savez_compressed(os.path.join(ARTIFACTS, "usf_mae_test_embeddings.npz"), features=F_, labels=Y, ids=te.Image_name.to_numpy())
            from sklearn.decomposition import PCA
            Z = PCA(2, random_state=SEED).fit_transform(F_)
            for c in range(6): plt.scatter(*Z[Y == c].T, s=6, label=CLASS_SHORT[c])
            plt.legend(markerscale=3); plt.title("USF-MAE fine-tuned [CLS] embeddings (PCA)"); plt.show()
            '''), REFS]


def nb05():
    return [header(5, "Model evaluation and comparison", "Load the saved test predictions of all three models and compare confusion matrices, per-class precision/recall/F1, calibration (with temperature fitted on validation), label efficiency, misclassifications, machine subgroups and paired statistical tests. Export machine-readable results for the website.",
                   "CPU is fine (minutes)", "`runs/*/test_predictions.npz` from notebooks 02–04", "`website_export/{evaluation,training_curves,label_efficiency}.json`"),
            code(SETUP),
            code('''
            from omari_us.evaluation import metrics as M
            import glob
            RUN_FOR = {}
            for mid in ["densenet161", "vit_b16", "usf_mae"]:
                cands = sorted(glob.glob(os.path.join(RUNS, f"{mid}_finetune_lf1_s0*")))
                if cands: RUN_FOR[mid] = cands[0]
            print(RUN_FOR)
            P = {m: dict(np.load(os.path.join(r, "test_predictions.npz"), allow_pickle=True)) for m, r in RUN_FOR.items()}
            assert P, "No runs found — train with notebooks 02–04 first (same ARTIFACTS folder)."
            '''),
            md("## 1 · Headline metrics with patient-cluster bootstrap CIs\nResampling *patients* (not images) respects within-patient correlation."),
            code('''
            rows = []
            for m, z in P.items():
                y, p = z["labels"].astype(int), z["probs"]
                r = M.classification_report(y, p.argmax(1)); ci = M.bootstrap_ci(y, p.argmax(1), groups=z["patients"], n_boot=1000 if MODE == "full" else 200)
                rel = M.reliability(p, y)
                rows.append({"model": m, "n": len(y), "accuracy": r["accuracy"], "acc 95% CI": np.round(ci["accuracy"], 3), "macro-F1": r["macro_f1"],
                             "F1 95% CI": np.round(ci["macro_f1"], 3), "balanced acc": r["balanced_accuracy"], "ECE": rel["ece"], "NLL": M.nll(p, y)})
            pd.DataFrame(rows).set_index("model").round(3)
            '''),
            md("## 2 · Confusion matrices and per-class metrics"),
            code('''
            fig, axes = plt.subplots(1, len(P), figsize=(5 * len(P), 4.4))
            axes = np.atleast_1d(axes)
            for ax, (m, z) in zip(axes, P.items()):
                cm = M.confusion(z["labels"].astype(int), z["probs"].argmax(1)); cmn = cm / cm.sum(1, keepdims=True).clip(1)
                ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
                for i in range(6):
                    for j in range(6): ax.text(j, i, f"{cmn[i,j]*100:.0f}", ha="center", va="center", color="white" if cmn[i, j] > .5 else "black", fontsize=8)
                ax.set_xticks(range(6), CLASS_SHORT, rotation=45); ax.set_yticks(range(6), CLASS_SHORT); ax.set_title(m); ax.set_xlabel("predicted"); ax.set_ylabel("true")
            plt.tight_layout(); plt.show()
            pc = {m: pd.DataFrame(M.classification_report(z["labels"].astype(int), z["probs"].argmax(1))["per_class"], index=CLASS_SHORT) for m, z in P.items()}
            pd.concat(pc, axis=1).round(3)
            '''),
            md("## 3 · Calibration\nTemperature is fitted on the **validation** predictions, then applied to test."),
            code('''
            fig, ax = plt.subplots(figsize=(5, 5)); ax.plot([0, 1], [0, 1], "k--", lw=1)
            for m, z in P.items():
                y, p = z["labels"].astype(int), z["probs"]
                v = np.load(os.path.join(RUN_FOR[m], "val_predictions.npz"), allow_pickle=True)
                T = M.fit_temperature(v["probs"], v["labels"].astype(int))
                b = [r for r in M.reliability(p, y)["bins"] if r["count"]]
                ax.plot([r["conf"] for r in b], [r["acc"] for r in b], "o-", label=f"{m} (ECE {M.reliability(p, y)['ece']:.3f} → {M.reliability(M.apply_temperature(p, T), y)['ece']:.3f} at T={T:.2f})")
            ax.set_xlabel("confidence"); ax.set_ylabel("accuracy"); ax.legend(fontsize=7); plt.show()
            '''),
            md("## 4 · Paired comparisons\nExact McNemar tests on the same test images, Holm-adjusted. Image-level pairing ignores patient clustering; treat p-values as approximate."),
            code('''
            ms = list(P); out = []
            for i in range(len(ms)):
                for j in range(i + 1, len(ms)):
                    a, b = P[ms[i]], P[ms[j]]; oa, ob = np.argsort(a["ids"]), np.argsort(b["ids"])
                    assert (a["ids"][oa] == b["ids"][ob]).all()
                    t = M.mcnemar(a["labels"][oa].astype(int), a["probs"][oa].argmax(1), b["probs"][ob].argmax(1)); out.append({"a": ms[i], "b": ms[j], **t})
            if out:
                for o, adj in zip(out, M.holm([o["p_value"] for o in out])): o["p_holm"] = adj
            pd.DataFrame(out)
            '''),
            md("## 5 · Domain shift by machine and the hardest errors"),
            code('''
            for m, z in P.items():
                if "machines" in z and len(set(z["machines"])) > 1:
                    print(m); display(pd.DataFrame(M.by_group(z["labels"].astype(int), z["probs"].argmax(1), z["machines"])).T.round(3))
            m0 = list(P)[0]; z = P[m0]
            conf = z["probs"].max(1); wrong = np.nonzero(z["probs"].argmax(1) != z["labels"])[0]
            print(f"{m0}: {len(wrong)} errors; most confident errors:")
            pd.DataFrame({"id": z["ids"][wrong], "true": [CLASS_SHORT[i] for i in z["labels"][wrong]], "pred": [CLASS_SHORT[i] for i in z["probs"][wrong].argmax(1)], "confidence": conf[wrong]}).sort_values("confidence", ascending=False).head(10)
            '''),
            md("## 6 · Label efficiency\nCollected from `summary.json` files of the label-fraction sweeps (notebooks 02–04)."),
            code('''
            S = [json.load(open(f)) for f in glob.glob(os.path.join(RUNS, "*", "summary.json"))]
            le = pd.DataFrame([{"model": s["config"]["model"], "regime": s["config"]["regime"], "fraction": s["config"]["label_fraction"], "seed": s["config"]["seed"], "macro_f1": s["test"]["macro_f1"]} for s in S])
            if le.fraction.nunique() > 1:
                g = le.groupby(["model", "regime", "fraction"]).macro_f1.agg(["mean", "min", "max"]).reset_index()
                for (m, rg), d in g.groupby(["model", "regime"]):
                    plt.plot(d.fraction, d["mean"], "o-", label=f"{m} ({rg})"); plt.fill_between(d.fraction, d["min"], d["max"], alpha=.15)
                plt.xscale("log"); plt.xlabel("fraction of training labels (by patient)"); plt.ylabel("test macro-F1"); plt.legend(); plt.show()
            else:
                print("Only one label fraction found; run the optional sweeps in notebooks 02–04.")
            '''),
            md("## 7 · Export for the website\nUses the same exporter as `scripts/evaluate.py`, so the website's numbers are exactly these."),
            code('''
            from omari_us.export import Exporter
            from omari_us.datasets import load_split_manifest
            cfg = {"output_dir": os.path.join(ARTIFACTS, "website_export"), "evaluation": {"runs": RUN_FOR, "bootstrap": 1000 if MODE == "full" else 200,
                   "label_efficiency_glob": os.path.join(RUNS, "*", "summary.json"), "export_test_probabilities": True}}
            ex = Exporter(cfg); ex.out.mkdir(parents=True, exist_ok=True)
            splits_df = pd.read_csv(SPLITS_CSV)
            print(ex.export_evaluation(splits_df), ex.export_curves(), ex.export_label_efficiency())
            if MODE == "smoke": print("SMOKE MODE — do not copy these files to the website.")
            '''), REFS]


def nb06():
    return [header(6, "Explainability laboratory", "Compute CAM, Grad-CAM and Grad-CAM++ for DenseNet-161 and transformer-appropriate attributions (token Grad-CAM, attention rollout, gradient-weighted rollout) for both ViTs; plug in a MetaCAM++ implementation if available; test faithfulness (deletion/insertion, cascading randomisation, perturbation stability); export genuine heatmaps for the website.",
                   "GPU recommended (attributions need backward passes); demo ≈10 min", "fine-tuned checkpoints from notebooks 02–04", "figures, `faithfulness.csv`, and (optionally) the full website bundle"),
            code(SETUP), code(DATA), code(LOAD_SPLITS),
            code('''
            import glob
            from omari_us.models import load_finetuned
            from omari_us.datasets import input_spec, build_transforms, denormalize
            from omari_us import explainability as X
            from PIL import Image
            MODELS = {}
            for mid in ["densenet161", "vit_b16", "usf_mae"]:
                c = sorted(glob.glob(os.path.join(RUNS, f"{mid}_finetune_lf1_s0*", "best.pt")))
                if c:
                    m, meta_ = load_finetuned(mid, c[0]); MODELS[mid] = (m.to(DEVICE).eval(), build_transforms(input_spec(mid), False))
            print("loaded:", list(MODELS))
            print("method compatibility:", {m: X.methods_for(m) for m in MODELS})
            '''),
            md(r"""
            ## 1 · Methods and their assumptions
            | Method | Needs | Class-specific | Here |
            |---|---|---|---|
            | CAM | conv features → GAP → linear | yes | DenseNet-161 (`model.act`, 2208×7×7) |
            | Grad-CAM | any (B,C,H,W) activation | yes | DenseNet any block; ViT via token reshape |
            | Grad-CAM++ | as Grad-CAM; exp-score closed form | yes | same |
            | Attention rollout | attention matrices | **no** | ViT-B/16, USF-MAE |
            | Gradient-weighted rollout | attention + gradients | yes | ViT-B/16, USF-MAE |
            | MetaCAM++ | — | — | reserved; supply an implementation below |

            For a GAP-linear head, Grad-CAM = ReLU(CAM)/Z exactly (verified in `tests/test_models_and_xai.py`).
            """),
            code('''
            row = splits[splits.split == "test"].iloc[0]
            img = Image.open(row.path).convert("L")
            target = int(row.label)
            cols = [(m, meth) for m in MODELS for meth in X.methods_for(m)]
            fig, axes = plt.subplots(1, len(cols) + 1, figsize=(2.6 * (len(cols) + 1), 3))
            axes[0].imshow(img.resize((224, 224)), cmap="gray"); axes[0].set_title(f"true: {CLASS_SHORT[target]}", fontsize=8)
            for ax, (m, meth) in zip(axes[1:], cols):
                model, tf = MODELS[m]; x = tf(img).unsqueeze(0).to(DEVICE)
                a = X.normalise(X.upsample(X.explain(meth, m, model, x, target)))
                ax.imshow(img.resize((224, 224)), cmap="gray"); ax.imshow(a, cmap="inferno", alpha=.55); ax.set_title(f"{m}\\n{meth}", fontsize=7)
            for a in axes: a.axis("off")
            plt.tight_layout(); plt.show()
            '''),
            md("## 2 · MetaCAM++ (user-supplied)\nThis cell registers your implementation. Nothing is implemented here; leave `METACAMPP_PATH` empty to skip."),
            code('''
            METACAMPP_PATH = ""  #@param {type:"string"}
            if METACAMPP_PATH:
                import importlib.util
                spec_ = importlib.util.spec_from_file_location("metacampp", METACAMPP_PATH); mod = importlib.util.module_from_spec(spec_); spec_.loader.exec_module(mod)
                # the module is expected to call omari_us.explainability.register_method("metacampp", models=(...))
                print("registered:", X.compatible("metacampp", "densenet161"), X.compatible("metacampp", "vit_b16"))
            else:
                print("MetaCAM++ not supplied — reserved slot only.")
            '''),
            md("## 3 · Faithfulness: deletion and insertion\nRemove (or reveal) pixels in attribution order, replacing them with a heavy blur. Compare with a random order."),
            code('''
            N_IMAGES = 30 if MODE == "full" else (2 if MODE == "smoke" else 8)
            test_rows = splits[splits.split == "test"].sample(N_IMAGES, random_state=SEED)
            recs = []
            for _, r in test_rows.iterrows():
                im = Image.open(r.path).convert("L")
                for m, (model, tf) in MODELS.items():
                    x = tf(im).unsqueeze(0).to(DEVICE)
                    for meth in X.methods_for(m):
                        a = X.explain(meth, m, model, x, int(r.label))
                        d = X.faithfulness.deletion_insertion(model, x, a, int(r.label), steps=10, mode="deletion")
                        ins = X.faithfulness.deletion_insertion(model, x, a, int(r.label), steps=10, mode="insertion")
                        recs.append({"model": m, "method": meth, "deletion_auc": d["auc"], "insertion_auc": ins["auc"]})
                    rnd = torch.rand(14, 14)
                    recs.append({"model": m, "method": "random", "deletion_auc": X.faithfulness.deletion_insertion(model, x, rnd, int(r.label), steps=10)["auc"],
                                 "insertion_auc": X.faithfulness.deletion_insertion(model, x, rnd, int(r.label), steps=10, mode="insertion")["auc"]})
            fa = pd.DataFrame(recs).groupby(["model", "method"]).agg(["mean", "std"]).round(3)
            fa.to_csv(os.path.join(ARTIFACTS, "faithfulness.csv")); fa
            '''),
            md("**Read:** lower deletion AUC and higher insertion AUC than `random` indicate the ranking carries information about *this model's* decision. They say nothing about clinical correctness."),
            md("## 4 · Sanity check: cascading randomisation (Adebayo et al., 2018)"),
            code('''
            if "densenet161" in MODELS:
                model, tf = MODELS["densenet161"]; x = tf(img).unsqueeze(0).to(DEVICE)
                groups = ["classifier", "features.norm5", "features.denseblock4", "features.denseblock3", "features.denseblock2"]
                res = X.faithfulness.cascading_randomisation(model, x, target, lambda mm: X.gradcam(mm, x, target, mm.act), groups)
                display(pd.DataFrame(res))
            for m in [k for k in MODELS if k != "densenet161"]:
                model, tf = MODELS[m]; x = tf(img).unsqueeze(0).to(DEVICE)
                groups = ["head", "norm", "blocks.11", "blocks.10", "blocks.9", "blocks.6"]
                res = X.faithfulness.cascading_randomisation(model, x, target, lambda mm: X.vitx.token_gradcam(mm, x, target), groups)
                print(m); display(pd.DataFrame(res))
            '''),
            md("## 5 · Perturbation stability\nSame image under the perturbation grid used by the website; rank correlation with the unperturbed map and change in the true-class probability."),
            code('''
            from omari_us.datasets import perturbations as PT
            g = np.asarray(img, dtype=np.float32) / 255.0
            rows = []
            for kind, param, label in PT.DEFAULT_GRID:
                pg = PT.apply(g, kind, param, seed=0)
                pim = Image.fromarray((pg * 255).astype(np.uint8))
                for m, (model, tf) in MODELS.items():
                    meth = "gradcam" if m == "densenet161" else "vit_gradcam"
                    x0, x1 = tf(img).unsqueeze(0).to(DEVICE), tf(pim).unsqueeze(0).to(DEVICE)
                    with torch.no_grad(): p0, p1 = [torch.softmax(model(x), -1)[0, target].item() for x in (x0, x1)]
                    rho = X.faithfulness.spearman(X.explain(meth, m, model, x0, target), X.explain(meth, m, model, x1, target))
                    rows.append({"perturbation": label, "model": m, "Δp(true)": p1 - p0, "map ρ": rho})
            pd.DataFrame(rows).pivot(index="perturbation", columns="model").round(3)
            '''),
            md("## 6 · Export the full website bundle\nWrites predictions, feature maps, attention, attributions, reconstructions and perturbation experiments with provenance (see `configs/export.yaml`)."),
            code('''
            EXPORT = False  #@param {type:"boolean"}
            if EXPORT and MODE != "smoke":
                from omari_us.export import Exporter
                from omari_us.utils import load_yaml
                cfg = load_yaml(os.path.join(REPO, "configs", "export.yaml"))
                cfg["output_dir"] = os.path.join(ARTIFACTS, "website_bundle")
                cfg["models"] = {m: {"checkpoint": sorted(glob.glob(os.path.join(RUNS, f"{m}_finetune_lf1_s0*", "best.pt")))[0]} for m in MODELS}
                ck = os.path.join(ARTIFACTS, "checkpoints", "usf_mae_100ep.pth")
                if "usf_mae" in cfg["models"] and os.path.exists(ck): cfg["models"]["usf_mae"]["pretrain_checkpoint"] = ck
                cfg["evaluation"]["runs"] = {m: os.path.dirname(v["checkpoint"]) for m, v in cfg["models"].items()}
                cfg["evaluation"].pop("curve_runs", None); cfg["evaluation"]["label_efficiency_glob"] = os.path.join(RUNS, "*", "summary.json")
                Exporter(cfg).run(splits, clean=True)
                print("Zip", cfg["output_dir"], "and unpack it into website/public/experiments/")
            '''), REFS]


def main():
    os.makedirs(OUT, exist_ok=True)
    specs = {
        "01_dataset_preparation.ipynb": nb01(), "02_densenet161.ipynb": nb02(), "03_vit_b16.ipynb": nb03(),
        "04_usf_mae.ipynb": nb04(), "05_model_evaluation.ipynb": nb05(), "06_explainability.ipynb": nb06(),
    }
    for name, cells in specs.items():
        nb = nbf.v4.new_notebook()
        nb.cells = cells
        nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"}, "language_info": {"name": "python"},
                       "accelerator": "GPU", "colab": {"provenance": [], "gpuType": "T4"}}
        nbf.validate(nb)
        nbf.write(nb, os.path.join(OUT, name))
        print("wrote", name, len(cells), "cells")


if __name__ == "__main__":
    main()
