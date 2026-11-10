# Data and model sources

## Asset status

| Asset | Source | Licence | In this repo? | Downloaded during the build? | How to obtain |
|---|---|---|---|---|---|
| FETAL_PLANES_DB (12,400 PNG + metadata CSV, 2.1 GB) | Zenodo 3904280 | CC BY 4.0 | No | **No** (zenodo.org blocked) | `bash scripts/download_assets.sh data` or notebook 01 |
| USF-MAE 100-epoch checkpoint | Google Drive `1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr` (linked from github.com/Yusufii9/USF-MAE) | research release, see repo | No | **No** (Drive blocked) | `scripts/download_assets.sh` (gdown) or notebook 04 |
| USF-MAE 500-epoch checkpoint (optional) | Google Drive `15n71qHXSBjEjrLtCOzvBb6gTsGhh7NUg` | same | No | No | gdown |
| USF-MAE code | github.com/Yusufii9/USF-MAE | MIT | No (re-implemented, key-compatible) | Cloned for inspection only | — |
| DenseNet-161 ImageNet weights | torchvision `DenseNet161_Weights.IMAGENET1K_V1` | BSD-3 code; ImageNet terms | No | **No** (download.pytorch.org blocked) | automatic on first `build_model("densenet161")` |
| ViT-B/16 ImageNet weights | timm `vit_base_patch16_224.augreg_in1k` (HF hub `timm/vit_base_patch16_224.augreg_in1k`) | Apache-2.0 | No | **No** (huggingface.co blocked) | automatic on first `build_model("vit_b16")` |
| Fine-tuned checkpoints (3 models × 2 regimes) | produced by notebooks 02–04 | yours | No | **Not trained** (no GPU, no data) | see `docs/TRAINING_PLAN.md` |
| Website artifact bundle | produced by `scripts/export_website_artifacts.py` | derived; CC BY 4.0 attribution for images | No (`website/public/experiments/` is empty) | Not produced | see `docs/TRAINING_PLAN.md` |

## Citations

- **Dataset.** Burgos-Artizzu X.P., Coronado-Gutiérrez D., Valenzuela-Alcaraz B., Bonet-Carne E., Eixarch E., Crispi F., Gratacós E. *Evaluation of deep convolutional neural networks for automatic classification of common maternal fetal ultrasound planes.* Scientific Reports 10, 10200 (2020). https://doi.org/10.1038/s41598-020-67076-5. Data: https://zenodo.org/records/3904280
- **DenseNet.** Huang G., Liu Z., van der Maaten L., Weinberger K.Q. *Densely connected convolutional networks.* CVPR 2017.
- **ViT.** Dosovitskiy A. et al. *An image is worth 16×16 words.* ICLR 2021. AugReg weights: Steiner A. et al. *How to train your ViT?* TMLR 2022.
- **MAE.** He K., Chen X., Xie S., Li Y., Dollár P., Girshick R. *Masked autoencoders are scalable vision learners.* CVPR 2022.
- **USF-MAE.** Megahed Y., Ducharme R., Erman A., Walker M.C., Hawken S., Chan A.D.C. *USF-MAE: Ultrasound self-supervised foundation model with masked autoencoding.* Biomedical Signal Processing and Control 122, 110313 (2026). arXiv:2510.22990.
- **CAM.** Zhou B. et al. CVPR 2016. **Grad-CAM.** Selvaraju R.R. et al. ICCV 2017. **Grad-CAM++.** Chattopadhay A. et al. WACV 2018.
- **Attention rollout.** Abnar S., Zuidema W. ACL 2020. **Transformer attribution.** Chefer H., Gur S., Wolf L. CVPR 2021.
- **Sanity checks.** Adebayo J. et al. NeurIPS 2018. **Deletion/insertion.** Petsiuk V. et al. BMVC 2018. **Calibration.** Guo C. et al. ICML 2017. **Effective-number weights.** Cui Y. et al. CVPR 2019. **Shortcut learning.** Geirhos R. et al. Nat. Mach. Intell. 2020.

## Initialisation and training summary (to fill in after training)

| | DenseNet-161 | ViT-B/16 | USF-MAE |
|---|---|---|---|
| Initialisation | torchvision IMAGENET1K_V1 | timm augreg_in1k | USF-MAE 100-ep encoder (← Facebook MAE IN-1k) |
| Pretraining data | ImageNet-1k (labelled) | ImageNet-1k (labelled) | ImageNet-1k (unlabelled MAE) → OpenUS-46 ≈370k US images (unlabelled, incl. FETAL_PLANES_DB) |
| Params (6-way) | 26,485,254 | 85,803,270 | 85,803,270 |
| Input | 224² stretch, bilinear, IN μ/σ | 224² stretch, bicubic, 0.5/0.5 | 224² stretch, nearest, IN μ/σ |
| Optimiser | AdamW, backbone lr ×0.1 | AdamW, layer decay 0.75 | AdamW, layer decay 0.75 |
| Augmentation | rot ±10°, scale 0.85–1, h-flip, brightness/contrast ±0.15 | same | same |
| Epochs / selection | ≤25, early stop on val macro-F1 | same | same |
| Compute (GPU-h) | *record* | *record* | *record* |
| Checkpoint sha256 | *record (summary.json)* | *record* | *record* |
