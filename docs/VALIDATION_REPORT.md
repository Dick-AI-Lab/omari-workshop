# Validation report

Build environment: Linux, Python 3.13, Node 22, torch 2.15 (CPU), 2 CPU cores, no GPU. Date: 4 November 2026.

## Executed and passed

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

## Not executed (blocked) and what it means

| Check | Why blocked | Consequence |
|---|---|---|
| Download FETAL_PLANES_DB | zenodo.org not reachable | No real images, no real dataset statistics. Website uses literature counts (labelled) |
| Download USF-MAE checkpoint | drive.google.com not reachable | Checkpoint container and decoder presence **unverified**. The loader handles the formats MAE produces and reports what it finds |
| Download DenseNet-161 / ViT-B/16 ImageNet weights | download.pytorch.org, huggingface.co not reachable | Pretrained initialisation not exercised here; only architectures and random init |
| Training on real data (any mode other than smoke) | no data, no weights, no GPU | **No model results exist.** All model-output figures show "Awaiting artifact" |
| `demo`/`full` notebook modes | same | Code paths identical to smoke except data source and `pretrained=True`; those two branches are untested here |
| Colab execution | not available | Setup cell logic (clone, pip, Drive) untested in Colab itself; `REPO_URL` must be set |
| Cross-browser / screen-reader testing | Chromium only | Keyboard focus styles and ARIA labels present; no assistive-technology test performed |

## Scientific results: status

None. Every number shown during the smoke runs came from random weights on synthetic images and was written only to `/tmp`. Nothing from those runs is in the repository.
