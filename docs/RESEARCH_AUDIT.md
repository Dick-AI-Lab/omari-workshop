# Research and technical audit

Phase 1 of the build: what was verified before implementation, how, and the decisions that followed.
Audit date: 4 November 2026. Every item is marked **verified** (checked against a primary source or by running code), **reported** (taken from a publication and not independently re-derived), or **unverified / blocked**.

## 1. Build environment

| Item | Finding |
|---|---|
| Compute | 2 CPU cores, ~7–8 GB RAM, **no GPU** |
| Network | PyPI, npm, GitHub (git and REST API) reachable. **Blocked:** zenodo.org, huggingface.co, download.pytorch.org, drive.google.com, storage.googleapis.com, CDNs |
| Consequence | FETAL_PLANES_DB, the USF-MAE checkpoint, torchvision DenseNet-161 weights and timm ViT weights **could not be downloaded here**. No model was trained on real data, and no authentic model output exists in this repository. |
| Installed for validation | torch 2.15 (CPU), torchvision 0.29, timm 1.0.31, scikit-learn 1.9, Node 22, Vite 5, React 18, D3 7, KaTeX 0.16 |

## 2. FETAL_PLANES_DB

| Fact | Status | Source |
|---|---|---|
| Zenodo record 3904280, file `FETAL_PLANES_ZENODO.zip`, 2.1 GB, md5 `2a5fcc2cefb789bcc0f6c1f73e0ea43f` | verified (record page) | https://zenodo.org/records/3904280 |
| Licence **CC BY 4.0** | verified (record page) | same |
| 12,400 images, 1,792 patients, 2 hospitals (BCNatal: Hospital Clínic, Hospital Sant Joan de Déu), labelled by an expert maternal–fetal clinician | reported | Burgos-Artizzu et al., Sci. Rep. 10:10200 (2020) |
| 6 classes: fetal abdomen, brain, femur, thorax; maternal cervix; other. Brain images sub-labelled trans-thalamic / trans-cerebellum / trans-ventricular | reported | same |
| Split: patients ordered chronologically by first visit; first half train (7,129 images, 896 patients), second half test (5,271 images, 896 patients); 10% of train held out for validation | reported | same (Methods) |
| Machines: 3× Voluson E6, Voluson S8, Voluson S10 (GE), Aloka; "several operators with similar experience" | reported | same |
| Released as uncompressed greyscale PNG, DICOM headers removed | reported | same |
| Metadata CSV `FETAL_PLANES_DB_data.csv`, **semicolon-separated**, columns `Image_name; Patient_num; Plane; Brain_plane; Operator; US_Machine; Train` (trailing space in `Train `), image names like `Patient00001_Plane1_1_of_15` | verified indirectly (printed output of a public third-party notebook) | github.com/alejandroribero1/clasificacion-planos-fetales-ultrasonido |
| Per-class counts: abdomen 711, brain 3,092, femur 1,040, thorax 1,718, cervix 1,626, other 4,213 | reported (Table 1, transcribed). Cross-checks: the sum is 12,400; "Other = 4,213" and the brain sub-plane total of 2,949 (1,543 train + 1,406 test) appear in the paper text | Notebook 01 recomputes these from the CSV and the website then prefers the computed values |
| Best model in the paper: **DenseNet-169** (not -161), top-1 error 6.2%, average class accuracy 93.6%; technicians 5.1% and 6.5% | reported | same |

**Implication.** The paper's numbers are not used as stand-ins for our models anywhere. The website shows the literature class counts only for the class-balance figure, labelled with their source, until notebook 01 exports `dataset_stats.json`.

## 3. USF-MAE

Repository inspected by cloning https://github.com/Yusufii9/USF-MAE (MIT licence).

| Fact | Status |
|---|---|
| Paper: Megahed, Ducharme, Erman, Walker, Hawken, Chan. *USF-MAE: Ultrasound self-supervised foundation model with masked autoencoding.* Biomed. Signal Process. Control 122:110313 (2026); arXiv:2510.22990 | verified (README, arXiv) |
| Architecture `mae.models_mae.mae_vit_base_patch16_dec512d8b`: ViT-B/16 encoder (768-d, 12 blocks, 12 heads), decoder 512-d, 8 blocks, 16 heads | verified (pretraining notebook imports) |
| **Initialisation:** the pretraining notebook loads `mae_checkpoint/mae_pretrain_vit_base_full.pth` (Facebook ImageNet-1k MAE) into the model before ultrasound pretraining | verified (notebook source) |
| Pretraining preprocessing (`US_data_loading.py`): `Resize((224,224), NEAREST)` without aspect preservation, ImageNet mean/std, RGB; augmentation: rotation 0–90°, H/V flips, RandomResizedCrop(scale 0.5–2.0) | verified |
| Optimisation: AdamW, lr ∈ {3e-4, 1e-3} × 128/256, weight decay ∈ {0.01, 0.05}, 100 epochs, warm-up + half-cosine, bf16 autocast, grad clip 1.0, 5-fold CV | verified |
| Pretraining corpus OpenUS-46 (~370k images, 46 datasets). **FETAL_PLANES_DB is dataset 25 of OpenUS-46** | verified (README table) |
| 100-epoch checkpoint: Google Drive id `1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr`; 500-epoch: `15n71qHXSBjEjrLtCOzvBb6gTsGhh7NUg` | verified (README links). **Download blocked here**, so checkpoint contents (container format, decoder presence) are **unverified** |
| Reported downstream results: BUS-BRA, MMOTU-2D, GIST514-DB (F1 81.6 / 79.6 / 82.4%). **No FETAL_PLANES_DB downstream result is reported** | verified (README, abstract) |
| `norm_pix_loss`: the builder default in the MAE code is `False`, and the notebook does not override it | verified by reading the call; the MAE source itself was not vendored |

**Implementation decision.** We do not vendor facebookresearch/mae (CC BY-NC 4.0). `omari_us/models/usf_mae.py` re-implements `MaskedAutoencoderViT` with identical parameter names and shapes on timm `Block`/`PatchEmbed`. Tests verify:
1. A checkpoint in MAE's `{'model': state_dict}` format, with `module.` prefixes, loads and is reported as complete.
2. The encoder transfers into `timm.vit_base_patch16_224`, and the MAE encoder output with 0% masking equals the timm ViT features to 1e-5.
3. Encoder-only checkpoints are accepted for classification and rejected for reconstruction.
4. The sin-cos positional table matches MAE's reference values.

## 4. Baseline weights

| Model | Choice | Why | Status |
|---|---|---|---|
| DenseNet-161 | torchvision `DenseNet161_Weights.IMAGENET1K_V1` (acc@1 77.138 as published by torchvision) | Requested model; official weights | API verified. Parameter count 28,681,000 (1000-way) / 26,485,254 (6-way) verified by instantiation. Weights not downloadable here |
| ViT-B/16 | timm `vit_base_patch16_224.augreg_in1k` | Supervised **ImageNet-1k only** (same pretraining corpus as DenseNet and as USF-MAE's MAE initialisation); same timm module layout as the USF-MAE encoder, so attention and attribution code is shared | Pretrained config verified via `timm.models.get_pretrained_cfg`: mean = std = 0.5, bicubic, crop_pct 0.9, Apache-2.0. Parameter count 85,803,270 (6-way) verified |
| Alternative ViT | `vit_base_patch16_224.augreg2_in21k_ft_in1k` or torchvision `vit_b_16` | Stronger or different-recipe baselines | Configurable via `configs/vit_b16.yaml: timm_name` |

ViT-B/16 here is the **supervised** baseline. MAE pretraining appears only under USF-MAE.

## 5. Explainability methods: compatibility decisions

| Method | DenseNet-161 | ViT-B/16 / USF-MAE | Basis |
|---|---|---|---|
| CAM | exact (GAP → linear head) | not applicable (no conv maps; [CLS] head) | Zhou et al. 2016 |
| Grad-CAM | any conv block; final layer = ReLU(CAM)/49, a unit-tested identity | on patch tokens at the input of `blocks[b].norm1`, reshaped 196 → 14×14 ([CLS] dropped) | Selvaraju et al. 2017; standard ViT adaptation (pytorch-grad-cam convention) |
| Grad-CAM++ | closed form under S = exp(y) | same token reshape | Chattopadhay et al. 2018 |
| Attention rollout | — | ½(Ā + I) products, not class-specific | Abnar & Zuidema 2020 |
| Gradient-weighted rollout | — | E_h[ReLU(∂y/∂A ⊙ A)] rollout | simplified from Chefer et al. 2021 (no LRP relevance propagation, so **not** a full re-implementation) |
| MetaCAM++ | reserved | reserved | Presenter's ongoing research; not implemented, no equations or results invented |

Timm's fused attention is disabled temporarily to materialise attention matrices. The attention hooks were verified (rows sum to 1; flag restored).

## 6. Distill.pub: what we took and what we did not

Pages read: *Communicating with Interactive Articles* (Hohman et al., 2020) and *The Building Blocks of Interpretability* (Olah et al., 2018). *Feature Visualization* (2017) is known to the authors but was not re-read during this audit.

**Adopted:**
- explanation and experiment in the same place (every lab sits inside the text it illustrates)
- details on demand (*Mathematical details*, *View PyTorch implementation* panels; hover provenance)
- prediction prompts before reveal (Chapters 1 and 6)
- linked views (one shared image drives every lab)
- multiple representations of one idea (toy model next to real maps)

The trust caveat from *Building Blocks* runs through Chapters 5 and 6: interfaces can mislead, attribution methods are unsettled, and a model can look right in an interface while being wrong.

**Deliberately not adopted:**
- heavy scroll-driven animation. Hohman et al. note that interactivity can distract and that "not everything needs to be interactive", so motion is limited to direct responses to user input.
- copying Distill's visual style or figures.

## 7. Licensing and reuse audit

| Asset | Licence | Use here |
|---|---|---|
| FETAL_PLANES_DB images and metadata | CC BY 4.0 | Not redistributed in the repo. Exported sample PNGs in `website/public/experiments/` are allowed with attribution (attribution is in the website footer and the manifest) |
| USF-MAE code | MIT | Not copied. Re-implemented model with compatible keys; checkpoint downloaded by the user |
| USF-MAE checkpoints | Not stated separately in the README; distributed for research | Not redistributed |
| facebookresearch/mae | CC BY-NC 4.0 | **Not vendored** (to avoid the NC restriction); only the published algorithm is re-implemented |
| torchvision weights | BSD-3 (code); ImageNet terms apply to weights | Downloaded at run time |
| timm `augreg_in1k` | Apache-2.0 (per timm config) | Downloaded at run time |
| Fonts (Source Serif 4, IBM Plex) | SIL OFL 1.1 | Bundled via @fontsource |
| KaTeX, D3, React | MIT / ISC | npm dependencies |

## 8. Resource requirements (estimates, not measurements)

- **Training.** Each model × regime (`full` mode, 25 epochs, ~6.4k training images) is roughly 0.5–2 GPU-hours on a T4. The label-efficiency sweep (6 fractions × 3 seeds × 3 models, frozen regime) adds about 54 short runs. These timings are extrapolated from typical ViT-B/DenseNet-161 throughput, not measured.
- **Export bundle.** ~20 samples with full attributions, perturbations and attention is estimated at 30–80 MB, and ~15–30 min on a GPU.
- **Website.** Production JS 614 kB minified (198 kB gzip); no runtime network dependencies except the optional local inference server.

## 9. Open questions for the presenter

1. **USF-MAE checkpoint container.** Does it include decoder weights? This determines whether authentic reconstructions can be shown.
2. **Reporting the FETAL_PLANES_DB pretraining overlap.** Should a FETAL_PLANES_DB-excluded USF-MAE variant be used, if one exists?
3. **Fine-tuning head for USF-MAE.** Token pooling (default; same as the supervised ViT) or MAE's average pooling + `fc_norm`?
4. **MetaCAM++.** Reference implementation and the information to show in Chapter 5.
