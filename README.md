<<<<<<< HEAD
# From Pixels to Predictions

**Developing AI models for medical image interpretation.** An interactive, research-grade two-hour workshop for the Ottawa Medical AI Research Institute (OMARI) Seminar Series.

> How do different approaches to learning image representations affect prediction, generalisation and explainability in prenatal ultrasound?

Six-class maternal–fetal plane classification on FETAL_PLANES_DB. The models are DenseNet-161 (supervised CNN), ViT-B/16 (supervised transformer) and USF-MAE (ultrasound self-supervised MAE foundation model).

## What is here

| Path | What it is |
|---|---|
| `website/` | The workshop itself: a Distill-inspired interactive article in React + TypeScript + Vite, with D3 and KaTeX. Seven chapters plus a break, presenter mode, and over 25 interactive figures |
| `notebooks/` | Six standalone Colab notebooks (`smoke` / `demo` / `full` modes) |
| `python/omari_us/` | Reusable package: data and splits, the three models (incl. a key-compatible USF-MAE), training, metrics, explainability, artifact export, live-inference server |
| `scripts/` | CLI: download, prepare, train, evaluate, export, provenance check, smoke test, notebook builder |
| `configs/` | Model recipes and export settings |
| `tests/` | pytest (Python) and Vitest (`website/src/lib/lib.test.ts`) |
| `docs/` | Research audit, instructor guide, storyboard, limitations, sources and licences, training plan, artifact schema, validation report |

## Run the website

```bash
cd website
npm install
npm run dev          # http://localhost:5173  (press P to present)
# or a static build that works offline:
npm run build && npm run preview   # http://localhost:4173 ; dist/ can be served from any static host
```

The site works **without any model artifacts**. Live simulations run, and every figure that needs a trained model says so and explains how to produce the output. Once you export a bundle into `website/public/experiments/`, those figures fill in automatically.

## Reproduce the experiments

```bash
pip install -r requirements.txt
bash scripts/download_assets.sh data                                  # FETAL_PLANES_DB + USF-MAE checkpoint
python scripts/prepare_dataset.py --data-root data --out artifacts/splits
python scripts/train.py --config configs/densenet161.yaml --regime finetune   # likewise vit_b16, usf_mae; frozen
python scripts/export_website_artifacts.py --config configs/export.yaml --clean
python scripts/check_provenance.py website/public/experiments
python -m omari_us.serve --config configs/export.yaml                 # optional live inference for the website
```

Full plan, with timings and optional experiments: `docs/TRAINING_PLAN.md`. In Colab, open the notebooks in order (01 → 06) and set `REPO_URL` in the first cell.

## Checks

```bash
python -m pytest -q tests                      # 22 tests: metrics, splits, perturbations, models, MAE checkpoint transfer, attribution identities
cd website && npm test && npm run typecheck   # 21 Vitest tests; strict TypeScript
python scripts/smoke_test_pipeline.py         # synthetic data + random weights → train → export → provenance check (to /tmp)
```

## Provenance rules

Every figure carries one of five badges: **Live simulation**, **Model output**, **Live inference**, **Illustrative**, **Awaiting artifact**. Model outputs are written only by `omari_us.export` from real checkpoints, with checkpoint hash, script and parameters. Nothing in this repository fabricates probabilities, heatmaps, reconstructions or metrics.

## Status

See `docs/VALIDATION_REPORT.md` for what was executed and what remains to be done. In short: the code, website, notebooks and pipeline are implemented and tested. **No model has been trained on real data yet** (the build environment had no GPU and could not reach Zenodo, Hugging Face, PyTorch or Google Drive).

## Licence

Code: MIT. FETAL_PLANES_DB: CC BY 4.0 (attribution required for exported images). See `LICENSE` and `docs/DATA_AND_MODEL_SOURCES.md`.
=======