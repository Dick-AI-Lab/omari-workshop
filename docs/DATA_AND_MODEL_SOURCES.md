# Data and model sources

*Updated 2026-11-14, after the Colab training runs (2026-11-10 to 2026-11-12). The pre-training version of this file said "not downloaded / not trained". That is no longer true: all assets below were obtained and used in Colab.*

## Asset status

| Asset | Source | Licence | In git? | Integrity (verified 2026-11-14) | Used for |
|---|---|---|---|---|---|
| FETAL_PLANES_DB (12,400 PNG + metadata CSV, 2.1 GB zip) | Zenodo 3904280, `https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1` | CC BY 4.0 | No (local copy in `content/data/FETAL_PLANES_ZENODO/`, ignored) | md5 `2a5fcc2cefb789bcc0f6c1f73e0ea43f` (matches the documented value) | All runs |
| **USF-MAE 500-epoch checkpoint (official for this workshop)** | Google Drive `15n71qHXSBjEjrLtCOzvBb6gTsGhh7NUg` (linked from github.com/Yusufii9/USF-MAE) | **Not stated**; see licences below | No (`content/omari_artifacts/omari_artifacts/checkpoints/usf_mae_500ep.pth`, 447,764,666 B) | sha256 `c659a06170b0f6967ae7f6d066681b8dea2c122a5f325bb8a82df11ffb3a2533` | Initialisation of all 19 USF-MAE runs; 500ep reconstructions (if regenerated) |
| USF-MAE 100-epoch checkpoint (historical) | Google Drive `1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr` | Not stated | No (`…/checkpoints/usf_mae_100ep.pth`, 447,746,224 B) | sha256 `f815c629878c17136985af9f4fdc81c2cfa02a94e4d992c026699957f75ccb66` | Only the MAE reconstructions in the original Colab bundle (kept as `reconstructions_legacy`) |
| USF-MAE code | github.com/Yusufii9/USF-MAE | MIT | No (re-implemented, key-compatible) | n/a | Reference only |
| DenseNet-161 ImageNet weights | torchvision `DenseNet161_Weights.IMAGENET1K_V1` | BSD-3 code; ImageNet terms for weights | No | Downloaded by torchvision in Colab | DenseNet init |
| ViT-B/16 ImageNet weights | timm `vit_base_patch16_224.augreg_in1k` | Apache-2.0 (per timm config) | No | Downloaded by timm in Colab | ViT init |
| Fine-tuned checkpoints (38 runs) | notebooks 02–04 | derived | No (`content/omari_artifacts/omari_artifacts/runs/*/best.pt`, 8.5 GB) | sha256 recorded in each `summary.json`; 37/38 match the file on disk (exception: `densenet161_finetune_lf1_s0`, see `docs/PROVENANCE.md`) | Evaluation, XAI |
| Run summaries / histories | notebooks 02–04 | derived | **Yes** (`runs/*/summary.json`, `history.json`) | n/a | Provenance of every metric |
| Website artifact bundle | `scripts/finalize_website_bundle.py` from the Colab outputs | derived; CC BY 4.0 attribution for images | No (`website/public/experiments/`, deployed separately) | `scripts/check_provenance.py` | Website |

The repository pins **both full checkpoint hashes** in this document and in `docs/PROVENANCE.md`. The run summaries record only a 12-character prefix (`sha256:c659a06170b0`).

## Licences and reuse notes

- **FETAL_PLANES_DB** is CC BY 4.0. The bundle includes 24 sample images and the misclassified thumbnails, with attribution in the manifest and the website footer.
- **USF-MAE checkpoints have no stated licence.** They were initialised from Facebook's ImageNet MAE weights (`mae_pretrain_vit_base_full.pth`), which are released under **CC BY-NC 4.0**, so the derived checkpoints **may inherit a non-commercial restriction**. We do not redistribute them. Users download them from the authors' links. Anyone reusing the fine-tuned USF-MAE weights should treat them as non-commercial research use only until the authors state otherwise.
- facebookresearch/mae code (CC BY-NC 4.0) is **not vendored**. `omari_us/models/usf_mae.py` is an independent re-implementation with compatible parameter names.
- torchvision ImageNet weights: ImageNet terms (non-commercial research). timm `augreg_in1k`: Apache-2.0 (reported by timm).

## Citations

- **Dataset.** Burgos-Artizzu X.P., Coronado-Gutiérrez D., Valenzuela-Alcaraz B., Bonet-Carne E., Eixarch E., Crispi F., Gratacós E. *Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes.* Scientific Reports 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5. Data: https://zenodo.org/records/3904280
- **DenseNet.** Huang G., Liu Z., van der Maaten L., Weinberger K.Q. *Densely connected convolutional networks.* CVPR 2017.
- **ViT.** Dosovitskiy A. et al. *An image is worth 16×16 words.* ICLR 2021. AugReg weights: Steiner A. et al. *How to train your ViT?* TMLR 2022.
- **MAE.** He K., Chen X., Xie S., Li Y., Dollár P., Girshick R. *Masked autoencoders are scalable vision learners.* CVPR 2022.
- **USF-MAE.** Megahed Y., Ducharme R., Erman A., Walker M.C., Hawken S., Chan A.D.C. *USF-MAE: Ultrasound self-supervised foundation model with masked autoencoding.* Biomedical Signal Processing and Control 122, 110313 (2026). arXiv:2510.22990. *The volume, article number and DOI have not been checked against the publisher page.*
- **CAM.** Zhou B. et al. CVPR 2016. **Grad-CAM.** Selvaraju R.R. et al. ICCV 2017. **Grad-CAM++.** Chattopadhay A. et al. WACV 2018.
- **Attention rollout.** Abnar S., Zuidema W. ACL 2020. **Transformer attribution.** Chefer H., Gur S., Wolf L. CVPR 2021. The workshop's `grad_rollout` method is a simplified gradient-weighted rollout *inspired by* this paper, without LRP.
- **Attention is not explanation.** Jain S., Wallace B.C. NAACL 2019.
- **Sanity checks.** Adebayo J. et al. NeurIPS 2018. **Deletion/insertion.** Petsiuk V. et al. BMVC 2018. **Calibration.** Guo C. et al. ICML 2017. **Effective-number weights.** Cui Y. et al. CVPR 2019. **Shortcut learning.** Geirhos R. et al. Nat. Mach. Intell. 2020.
- **Interactive articles.** Hohman F. et al. *Communicating with interactive articles.* Distill 2020. Olah C. et al. *The building blocks of interpretability.* Distill 2018.

## Initialisation and training summary (as executed)

| | DenseNet-161 | ViT-B/16 | USF-MAE |
|---|---|---|---|
| Initialisation | torchvision IMAGENET1K_V1 | timm augreg_in1k | **USF-MAE 500-ep encoder** (`usf_mae_500ep.pth`, sha256 `c659a061…b3a2533`) ← Facebook MAE IN-1k |
| Pretraining data | ImageNet-1k (labelled) | ImageNet-1k (labelled) | ImageNet-1k (unlabelled MAE) → OpenUS-46 ≈370k US images (unlabelled, **includes FETAL_PLANES_DB**) |
| Params (6-way) | 26,485,254 | 85,803,270 | 85,803,270 |
| Input | 224² stretch, bilinear, IN μ/σ | 224² stretch, bicubic, 0.5/0.5 | 224² stretch, nearest, IN μ/σ |
| Optimiser (finetune) | AdamW lr 3e-4, backbone ×0.1, wd 1e-4, warm-up 1 | AdamW lr 2e-4, layer decay 0.75, wd 0.05, warm-up 3 | same as ViT |
| Augmentation | rot ±10°, scale 0.85–1, h-flip, brightness/contrast ±0.15 | same | same |
| Epochs / selection | ≤25, early stop (patience 6) on val macro-F1 | same | same |
| Runs executed | 17 (frozen 1, finetune 16) | 2 (frozen 1, finetune 1) | 19 (frozen 1, finetune 18) |
| Compute | **5.14 GPU-hours measured for all 38 runs** (sum of `history.json` epoch times; Colab T4). Full-data runs took 14–25 min each | | |
| Headline checkpoint sha256 (`summary.json`) | `20d40d72c51bf892a499edfc835eb3a14af9e3e9dd485af46006df4029fa72f8` (file on disk is now `456847049ab2…`, see PROVENANCE) | `49fe4168a8922632d88aa36fa1e85c55dc14db32d4b9159549c99fadbe684d8c` | s0 `65680650b75f30ff7f38d88afa27c820de87a20ebc3e7c42d761a9b5df9c29d1` |

The GPU type is not recorded in the artifacts. The notebook outputs show a Colab T4 with `torch 2.14.1+cu130`.
