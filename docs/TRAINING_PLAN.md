# What to train, and what to send back

Training could not run in the build environment: no GPU, and the dataset and pretrained weights were unreachable. This is the minimum set of runs that fills every model-output figure, then the optional runs. Everything runs on one GPU (a Colab T4 works; an A100 is ~4× faster).

## 0. One-time setup

```bash
git clone <this repo> omari-workshop && cd omari-workshop
pip install -r requirements.txt
bash scripts/download_assets.sh data            # FETAL_PLANES_DB + USF-MAE 100-ep checkpoint, with checks
python scripts/prepare_dataset.py --data-root data --out artifacts/splits
```
`prepare_dataset.py` prints patients per split, which must be disjoint. It also writes `website/public/experiments/dataset_stats.json`, after which the website uses real counts.

## 1. Required runs (6 runs, ≈4–8 GPU-hours)

```bash
for m in densenet161 vit_b16 usf_mae; do
  python scripts/train.py --config configs/$m.yaml --regime frozen
  python scripts/train.py --config configs/$m.yaml --regime finetune
done
```
Each run writes `runs/<model>_<regime>_lf1_s0/{best.pt, history.json, summary.json, test_predictions.npz, val_predictions.npz}`.

## 2. Recommended: label-efficiency sweep (frozen regime, ≈54 short runs)

```bash
for m in densenet161 vit_b16 usf_mae; do python scripts/train.py --config configs/$m.yaml --regime frozen --all-fractions; done
```
Fractions 1, 5, 10, 25, 50 and 100% of training **patients** × seeds 0–2. The website plots mean and min–max bands.

## 3. Export the website bundle

```bash
python scripts/export_website_artifacts.py --config configs/export.yaml --clean
python scripts/check_provenance.py website/public/experiments
cd website && npm run build
```
Set `samples.opening` in `configs/export.yaml` to pick the Chapter 1 image (any test `Image_name`).

## 4. Optional, scientifically valuable

| Run | Why |
|---|---|
| USF-MAE with `global_pool: avg` | MAE's own fine-tuning head; check that conclusions do not depend on the head |
| Leave-one-machine-out (train without Aloka, test on Aloka) | A stronger domain-shift test than by-machine subgroups |
| USF-MAE pretrained **without** FETAL_PLANES_DB (if the authors can provide one) | Removes the transductive pretraining overlap |
| Second ViT baseline `vit_base_patch16_224.augreg2_in21k_ft_in1k` | Shows the effect of a larger supervised pretraining corpus |

## What to send back if you want me to integrate the results

Either the whole `website/public/experiments/` folder (zip) after step 3, or the `runs/` folders plus `artifacts/splits/split_manifest.csv` and the USF-MAE checkpoint path, so the export can be run here. The pipeline needs no other change. If the USF-MAE checkpoint lacks decoder weights, reconstructions are skipped automatically and the website keeps its empty state for that panel.
