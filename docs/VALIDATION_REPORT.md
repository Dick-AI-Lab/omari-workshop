# Validation report

*Updated 2026-11-14. Part A covers post-training validation of the real Colab runs. Part B is the original pre-training build report from 2026-11-04. It is kept unchanged except for its headings, and it describes the build sandbox only.*

## Part A. Post-training validation (2026-11-14)

Source: `docs/POST_TRAINING_AUDIT_2026-11-14.md`. All checks were read-only, recomputed with `python -I` from `content/omari_artifacts/omari_artifacts/` (ART) on a local machine (Python 3.12.7, torch 2.6.0+cu124, timm 1.0.19, numpy 1.26, pandas 2.2, sklearn 1.5.1).

### A1. What is now resolved from Part B's "blocked" list

| Part B item | Status now |
|---|---|
| Download FETAL_PLANES_DB | Done in Colab. md5 `2a5fcc2cefb789bcc0f6c1f73e0ea43f` verified; 12,400 images, 1,792 patients |
| USF-MAE checkpoint container / decoder | Both `usf_mae_100ep.pth` and `usf_mae_500ep.pth` contain a full encoder and decoder (254 tensors). Full sha256 values are in `docs/DATA_AND_MODEL_SOURCES.md` |
| ImageNet weights | Downloaded and used in Colab |
| Training on real data | **38 runs** (MODE=`full`, T4), 5.14 GPU-hours |
| `demo` / `full` notebook modes, Colab execution | All six notebooks executed in `full` mode with no tracebacks. The executed notebooks are committed as the historical record |
| "No model results exist" | Superseded. Results below |

### A2. Checks on the real artifacts

| Check | Result |
|---|---|
| Split protocol | Authors' official Train/Test kept; val = 10% of Train, patient-disjoint (StratifiedGroupKFold, seed 42). Sizes train 6,425 / 809 patients, val 704 / 87, test 5,271 / 896. **Patient overlap 0** across all splits |
| Metric recomputation (all 38 runs) | Accuracy, macro-F1, balanced accuracy, per-class P/R/F1 and confusion matrices from `test_predictions.npz` **match `summary.json` exactly**. Val macro-F1 equals `best_val_macro_f1`; `best_epoch` = argmax of val F1 in all runs |
| Test-set isolation | No test-based selection found; early stopping and checkpoint choice use val only |
| Checkpoint integrity | sha256(`best.pt`) matches `summary.json` for **37/38** runs. Exception: `densenet161_finetune_lf1_s0`, whose file was overwritten by an interrupted rerun (see `docs/PROVENANCE.md`) |
| `evaluation.json` vs runs | Confusion, accuracy, F1 and per-class values match; ECE/NLL match to ~1e-8; patient-cluster bootstrap CIs and McNemar tests recompute identically |
| Label-efficiency grid | Every available cell matches its `summary.json`; missing cells (ViT < 100%, DenseNet/ViT lf1 s1–s2) are genuinely not run |
| Original bundle (`ART/website_bundle`) | All 3,109 referenced files exist; schema 1.0. Defects (fixed by `scripts/finalize_website_bundle.py`): 1-point frozen label-efficiency curve, USF-MAE overlap note dropped, near-duplicate stats dropped, reconstructions from the 100ep checkpoint, `chefer` method label |
| Original `ART/website_export/label_efficiency.json` | Contains bare `NaN` (invalid JSON). Not used |

### A3. Verified results

Test set, fine-tuned, 100% labels (patient-cluster bootstrap, 1,000 resamples):

| Model | Seeds | Accuracy | Macro-F1 | 95% CI macro-F1 |
|---|---|---|---|---|
| DenseNet-161 | 1 | 0.9474 | 0.9425 | [0.9353, 0.9489] |
| ViT-B/16 | 1 | 0.9465 | 0.9410 | [0.9334, 0.9480] |
| USF-MAE (500ep) | 3 | 0.9471 ± 0.0021 | 0.9416 ± 0.0023 | s0: [0.9331, 0.9465] |

Exact McNemar tests give p of 0.741 to 1.0, and Holm-adjusted p = 1 for every pair, so the models are **statistically indistinguishable**. Calibration: ECE 0.13–0.14 before temperature scaling. The models are under-confident (fitted T ≈ 0.57–0.62), and ECE after temperature scaling is about 0.01. Frozen-probe macro-F1 is 0.7657 / 0.8038 / 0.8039 (see the frozen caveat in `docs/TRAINING_PLAN.md`). Label efficiency is in the README.

### A4. Not yet validated

- A website build with the finalized bundle installed, and a visual check of every chapter. This is pending until `scripts/finalize_website_bundle.py` has run and `npm ci && npm run build` succeeds.
- pytest and Vitest were not re-run as part of the post-training audit.
- The exact label-fraction subsets are not saved. A re-implementation did not reproduce the image counts.
- Bit-for-bit reproducibility (training is not deterministic).

## Part B. Pre-training build report (2026-11-04, historical)

Build environment: Linux, Python 3.13, Node 22, torch 2.15 (CPU), 2 CPU cores, no GPU. Date: 4 November 2026.

### Executed and passed (build sandbox, synthetic data)

| Check | Command | Result |
|---|---|---|
| TypeScript strict type check | `cd website && npm run typecheck` | pass |
| Website unit tests (Vitest) | `npm test` | **21/21 pass**. Architecture arithmetic matches torchvision/timm (DenseNet-161 28,681,000 / 26,485,254; ViT-B/16 86,567,656 / 85,803,270), MAE sin-cos values, masking count, convolution, letterbox/stretch, ImageNet normalisation, speckle moments, metrics, prior shift, ECE, ROC, Grad-CAM = ReLU(CAM)/Z, Grad-CAM++ closed form, leakage expectation |
| Production build | `npm run build` | pass (JS 614 kB / 198 kB gzip) |
| Python tests | `python -m pytest -q tests` | **22/22 pass**. Metrics (incl. temperature recovery, cluster bootstrap wider than i.i.d., McNemar, Holm); metadata parsing (semicolon CSV, trailing spaces); patient-disjoint splits; patient-level label fractions; duplicate detection; perturbation semantics = website semantics; model parameter counts; **USF-MAE checkpoint round-trip: MAE encoder output == timm ViT features (atol 1e-5)**; encoder-only checkpoints rejected for reconstruction; seeded reconstruction masks; **DenseNet Grad-CAM == ReLU(CAM)/49 on a real network**; Grad-CAM++ / layer shapes; all ViT methods (14×14, finite); attention rows sum to 1; fused-attention flag restored; incompatible and reserved methods refused; deletion curve |
| End-to-end pipeline smoke test | `python scripts/smoke_test_pipeline.py` | pass. Synthetic images → splits → demo training of all three models (random init; USF-MAE from a random MAE checkpoint) → export of predictions, feature maps, attention (incl. uint16 matrices), all attribution methods, MAE reconstructions, perturbations, evaluation, curves, label efficiency → `check_provenance.py` pass |
| Website with a populated bundle | smoke bundle served from a throwaway copy of `dist/`, inspected with Playwright | all model-output figures render; no console errors (after fixing three chart-domain bugs found this way) |
| Live-inference server | `python -m omari_us.serve` on smoke checkpoints | `/health`, `/predict` (probabilities sum to 1), `/explain` (7×7 DenseNet, 14×14 ViT); CAM on ViT correctly refused (422) |
| Website ↔ server | Playwright: change contrast to an unexported value → "Run live inference" | predictions shown with "Live inference" badge; stale results cleared on change (bug found and fixed) |
| Notebook syntax | AST parse of every code cell | pass (58 code cells) |
| **Notebook execution** | nbclient, `OMARI_MODE=smoke`, notebooks 01 → 06 in order | **6/6 pass** (01: 9 s, 02: 30 s, 03: 43 s, 04: 65 s, 05: 5 s, 06: 169 s). Notebook 06 initially ran out of memory: attribution backward passes left parameter gradients allocated. Fixed by computing input-only gradients (`no_param_grads`); peak memory fell from >5 GB to 3.5 GB |
| Provenance check on the shipped (empty) bundle folder | `python scripts/check_provenance.py website/public/experiments` | pass ("no bundle: website shows empty states") |
| Visual review | Playwright screenshots of all 28 figures, presenter mode, 820 px tablet width | layout issues found and fixed (wide figures constrained to the text column, chapter head alignment, title offset, toy feature channels, label overlap) |

### Not executed at the time (blocked), status as of 2026-11-14 in the post-training section above

| Check | Why blocked | Consequence |
|---|---|---|
| Download FETAL_PLANES_DB | zenodo.org not reachable | No real images, no real dataset statistics. Website uses literature counts (labelled) |
| Download USF-MAE checkpoint | drive.google.com not reachable | Checkpoint container and decoder presence **unverified**. The loader handles the formats MAE produces and reports what it finds |
| Download DenseNet-161 / ViT-B/16 ImageNet weights | download.pytorch.org, huggingface.co not reachable | Pretrained initialisation not exercised here; only architectures and random init |
| Training on real data (any mode other than smoke) | no data, no weights, no GPU | **No model results exist.** All model-output figures show "Awaiting artifact" |
| `demo`/`full` notebook modes | same | Code paths identical to smoke except data source and `pretrained=True`; those two branches are untested here |
| Colab execution | not available | Setup cell logic (clone, pip, Drive) untested in Colab itself; `REPO_URL` must be set |
| Cross-browser / screen-reader testing | Chromium only | Keyboard focus styles and ARIA labels present; no assistive-technology test performed |

### Scientific results: status at the time (superseded)

None. Every number shown during the smoke runs came from random weights on synthetic images and was written only to `/tmp`. Nothing from those runs is in the repository.
