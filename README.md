# From Pixels to Predictions

**Developing AI models for medical image interpretation.** An interactive, research-grade two-hour workshop for the Ottawa Medical AI Research Institute (OMARI) Seminar Series.

> How do different approaches to learning image representations affect prediction, generalisation and explainability in prenatal ultrasound?

The task is six-class maternal–fetal plane classification on FETAL_PLANES_DB (Burgos-Artizzu et al., 2020; 12,400 images, 1,792 patients). Three models are compared:

- **DenseNet-161**: supervised CNN, ImageNet-1k initialisation (torchvision `IMAGENET1K_V1`)
- **ViT-B/16**: supervised transformer, ImageNet-1k initialisation (timm `vit_base_patch16_224.augreg_in1k`)
- **USF-MAE**: ViT-B/16 encoder pretrained with masked autoencoding on unlabelled ultrasound (OpenUS-46)

## Status (2026-11-14)

All six notebooks were run in Colab (MODE=`full`, T4 GPU, 2026-11-10 to 2026-11-12). That produced 38 training runs, about **5.14 GPU-hours** measured. Every reported metric recomputes exactly from the saved prediction files. A post-training audit is in [`docs/POST_TRAINING_AUDIT_2026-11-14.md`](docs/POST_TRAINING_AUDIT_2026-11-14.md), and per-artifact provenance is in [`docs/PROVENANCE.md`](docs/PROVENANCE.md).

**USF-MAE checkpoint decision.** The official USF-MAE model for this workshop is the **500-epoch** checkpoint `usf_mae_500ep.pth` (sha256 `c659a06170b0f6967ae7f6d066681b8dea2c122a5f325bb8a82df11ffb3a2533`). All 19 USF-MAE runs were initialised from it. The repository originally specified the 100-epoch checkpoint (`usf_mae_100ep.pth`, sha256 `f815c629878c17136985af9f4fdc81c2cfa02a94e4d992c026699957f75ccb66`). That checkpoint is now **historical only**: it produced the MAE reconstructions in the original Colab bundle, which the website keeps as clearly labelled legacy items.

## Results (test set: 5,271 images, 896 patients)

Fine-tuned at 100% of the training labels. Models were selected on validation macro-F1, and the test set was never used for selection. The CIs are a patient-cluster bootstrap (1,000 resamples).

| Model | Seeds | Test accuracy | Test macro-F1 | 95% CI macro-F1 (seed 0) |
|---|---|---|---|---|
| DenseNet-161 | 1 | 0.9474 | 0.9425 | [0.9353, 0.9489] |
| ViT-B/16 | 1 | 0.9465 | 0.9410 | [0.9334, 0.9480] |
| USF-MAE (500ep init) | 3 | 0.9471 ± 0.0021 | 0.9416 ± 0.0023 | [0.9331, 0.9465] |

USF-MAE values are mean ± sample sd (ddof = 1) over seeds 0/1/2. By seed, macro-F1 is 0.9403 / 0.9402 / 0.9443 and accuracy is 0.9463 / 0.9456 / 0.9495.

**At full labels the three models are statistically indistinguishable.** Every exact McNemar test gives a Holm-adjusted p of 1, and the CIs overlap. This table does not support ranking the models. With a linear probe (the "frozen" regime, 1 seed), macro-F1 is DenseNet 0.7657, ViT 0.8038 and USF-MAE 0.8039. See the frozen-probe caveat in `docs/TRAINING_PLAN.md`.

**Label efficiency (fine-tune, test macro-F1, mean ± sd over 3 seeds).** The sweep subsamples training *patients*. We report macro-F1 rather than accuracy because 34% of images are "Other".

| Labelled fraction (train patients) | DenseNet-161 | USF-MAE | ViT-B/16 |
|---|---|---|---|
| 1% (10) | 0.563 ± 0.073 | 0.637 ± 0.031 | not run |
| 5% (41) | 0.808 ± 0.023 | 0.888 ± 0.008 | not run |
| 10% (82) | 0.882 ± 0.007 | 0.901 ± 0.015 | not run |
| 25% (203) | 0.923 ± 0.002 | 0.930 ± 0.005 | not run |
| 50% (406) | 0.933 ± 0.007 | 0.935 ± 0.005 | not run |
| 100% (809) | 0.9425 (1 seed) | 0.9416 ± 0.0023 | 0.9410 (1 seed) |

USF-MAE's advantage is clear only at 5% labels (the seed ranges do not overlap). It is noisy at 1%, small at 10% and negligible from 25% up. Two caveats apply. First, FETAL_PLANES_DB images (without labels) were in USF-MAE's pretraining corpus, which is transductive exposure, not label leakage. Second, no ViT-B/16 sweep was run, so this curve cannot separate ultrasound MAE pretraining from the ViT architecture. Read [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md) before presenting any of these numbers.

## What is here

| Path | What it is |
|---|---|
| `website/` | The workshop itself: a Distill-inspired interactive article in React + TypeScript + Vite, with D3 and KaTeX. Seven chapters plus a break, presenter mode, over 25 interactive figures. `website/src/data/` holds the artifact loader, types and literature values (source code, tracked) |
| `notebooks/` | Six Colab notebooks (`smoke` / `demo` / `full` modes). The committed copies are the **executed historical record** of the runs above |
| `python/omari_us/` | Reusable package: data and splits, the three models (including a key-compatible USF-MAE), training, metrics, explainability, artifact export, live-inference server |
| `scripts/` | CLI: download, prepare, train, evaluate, export, finalize website bundle, provenance check, smoke test, notebook generator |
| `configs/` | Model recipes and export settings |
| `tests/` | pytest (Python) and Vitest (`website/src/lib/lib.test.ts`) |
| `docs/` | Provenance, limitations, sources and licences, training plan (planned vs executed), artifact schema, validation report, research audit, instructor guide, storyboard, post-training audit |
| `content/` | **Local, mostly git-ignored** copy of the Colab filesystem (see below) |

## Setup

```bash
pip install -r requirements.txt
pip install -e python            # makes `omari_us` importable (alternatively: export PYTHONPATH=python)
```

Python 3.9+ is required, and torch ≥ 2.3 is recommended (the training loop uses `torch.amp.GradScaler`). The original runs used Colab torch 2.14.1+cu130 with timm unpinned. Local re-analysis used torch 2.6.0+cu124 and timm 1.0.19. Dependency versions are lower bounds only; no lock file exists for Python.

## Where the Colab outputs live

The Colab notebooks wrote to Google Drive (`/content/drive/MyDrive/omari_artifacts`). A copy of that filesystem is kept locally under `content/`:

```
content/data/FETAL_PLANES_ZENODO/            dataset (zip + 12,400 PNGs + CSV), git-ignored
content/omari_artifacts/omari_artifacts/     ART: the run outputs (note the double folder from the Drive zip)
  checkpoints/usf_mae_{100,500}ep.pth        pretrained USF-MAE (git-ignored)
  runs/<model>_<regime>_lf<frac>_s<seed>/    best.pt, summary.json, history.json, {val,test}_predictions.npz
  splits/split_manifest.csv, near_duplicates.csv
  dataset_stats.json, faithfulness.csv, usf_mae_test_embeddings.npz
  website_bundle/                            original nb06 bundle (schema 1.0); source for the corrected bundle
  website_export/                            earlier nb05 export (superseded; contains NaN)
```

Only small text provenance files under `content/` are tracked: `dataset_stats.json`, `faithfulness.csv`, `splits/*.csv`, and each run's `summary.json` and `history.json`. See `.gitignore`. Everything under `content/` is treated as **read-only**. The paths in `split_manifest.csv` are absolute Colab paths, so pass `--data-root content/data/FETAL_PLANES_ZENODO` to the CLIs to re-root them.

## Build the website bundle from the Colab outputs

The original `website_bundle/` has known defects: a one-point label-efficiency curve, the overlap note dropped, near-duplicate stats missing, and the method name `chefer`. Do not copy it by hand. Build the corrected bundle (schema 1.1) instead:

```bash
python scripts/finalize_website_bundle.py \
    --artifacts content/omari_artifacts/omari_artifacts \
    --out website/public/experiments
# optional: regenerate MAE reconstructions from the official 500ep checkpoint (GPU recommended)
python scripts/finalize_website_bundle.py --artifacts content/omari_artifacts/omari_artifacts \
    --out website/public/experiments --reconstruct --device cuda
python scripts/check_provenance.py website/public/experiments
```

`website/public/experiments/` is git-ignored except for its README. The bundle is about 70 MB and contains CC BY 4.0 dataset images, so ship it with the deployment rather than committing it. See [`website/public/experiments/README.md`](website/public/experiments/README.md).

## Run the website

```bash
cd website
npm ci
npm run dev                         # http://localhost:5173  (press P to present)
# static build that works offline:
npm run build && npx vite preview   # http://localhost:4173 ; dist/ can be served from any static host
```

The site works **without any model artifacts**. Live simulations run, and every figure that needs a trained model shows an "Awaiting artifact" state. Once the bundle is in `website/public/experiments/`, those figures fill in automatically (rebuild after adding it for a static deployment).

Optional live inference for the website. This needs `pip install -e "python[serve]"` and a config whose checkpoint paths point at existing `best.pt` files, for example under `content/omari_artifacts/omari_artifacts/runs/`:

```bash
python -m omari_us.serve --config configs/export.yaml --port 8000
```

## Reproduce the experiments from scratch

```bash
bash scripts/download_assets.sh data                                  # FETAL_PLANES_DB (md5-checked) + USF-MAE checkpoint
python scripts/prepare_dataset.py --data-root data --out artifacts/splits
python scripts/train.py --config configs/densenet161.yaml --regime finetune   # likewise vit_b16, usf_mae; --regime frozen
python scripts/evaluate.py --config configs/export.yaml --data-root data
```

`docs/TRAINING_PLAN.md` lists the planned grid and the grid that was actually executed. Training is seeded but not bit-deterministic (cuDNN non-deterministic, unseeded DataLoader workers, AMP). The label-fraction training subsets were not saved.

In Colab, open the notebooks in order (01 → 06). **The committed notebooks are the executed historical record and must not be regenerated over.** `scripts/build_notebooks.py` produces fresh, unexecuted copies. Write them elsewhere with `--out-dir` (it refuses to overwrite executed notebooks unless `--force` is given).

## Checks

```bash
python -m pytest -q tests                     # metrics, splits, perturbations, models, MAE checkpoint transfer, attribution identities, export
cd website && npm test && npm run typecheck   # Vitest; strict TypeScript
python scripts/smoke_test_pipeline.py --out <tmp dir>   # synthetic data + random weights → train → export → provenance check
```

## Provenance rules

Every figure carries one of five badges: **Live simulation**, **Model output**, **Live inference**, **Illustrative**, **Awaiting artifact**. Model outputs come only from real checkpoints, recorded with checkpoint hash, script and parameters. Nothing in this repository fabricates probabilities, heatmaps, reconstructions or metrics. `docs/PROVENANCE.md` traces every artifact the website uses back to its run, checkpoint and generator.

## Limitations (summary)

Full list: [`docs/SCIENTIFIC_LIMITATIONS.md`](docs/SCIENTIFIC_LIMITATIONS.md).

- **No ranking at full labels.** The differences are within noise. Seeds are unequal (DenseNet and ViT have 1 seed each, USF-MAE has 3).
- **Pretraining overlap.** USF-MAE's self-supervised corpus included FETAL_PLANES_DB images, test images included, without labels. This is transductive exposure, not label leakage, and it may favour USF-MAE, especially at low label budgets.
- **Confounded comparison.** Architecture, pretraining data and objective, and fine-tuning recipe all differ between the models. No ViT label-efficiency sweep was run.
- **Scanner shift.** Voluson S10 makes up 1,071 test images but only 11 training images and 0 validation images. Validation has only 87 patients.
- **Near-duplicates.** A loose dHash detector finds 2,685 train/val↔test pairs, all between different patients. There is no visible accuracy inflation.
- **DenseNet checkpoint.** `runs/densenet161_finetune_lf1_s0/best.pt` was overwritten by an interrupted rerun. The reported metrics come from the original run. See `docs/PROVENANCE.md`.
- **Frozen probe.** It was not strictly frozen: BatchNorm statistics updated during training.
- Single public dataset (two hospitals). Nothing here supports clinical use.

## Licence

Code: MIT. FETAL_PLANES_DB: CC BY 4.0 (attribution required for exported images). The USF-MAE checkpoints have no stated licence and may inherit CC BY-NC 4.0 from the Facebook MAE weights they were initialised from. See `LICENSE` and `docs/DATA_AND_MODEL_SOURCES.md`.
