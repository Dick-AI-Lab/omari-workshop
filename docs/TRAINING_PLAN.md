# Training plan: planned vs executed

*Updated 2026-11-14. Sections 1–2 compare the original pre-training plan with what actually ran in Colab between 2026-11-10 and 2026-11-12. The original plan, written when the build environment had no GPU, is kept in section 5 for reference.*

## 1. What was actually executed

All runs used Colab (T4 GPU, `torch 2.14.1+cu130`, MODE=`full`) through notebooks 02–04. The artifacts were written to Drive, and a copy is in `content/omari_artifacts/omari_artifacts/runs/`. The hyper-parameters in every `summary.json` match `configs/*.yaml`.

| Model | Regime | Label fractions | Seeds | Runs | Initialisation |
|---|---|---|---|---|---|
| DenseNet-161 | frozen | 1.0 | 0 | 1 | torchvision IMAGENET1K_V1 |
| DenseNet-161 | finetune | 1.0 | 0 | 1 | same |
| DenseNet-161 | finetune | 0.01, 0.05, 0.1, 0.25, 0.5 | 0, 1, 2 | 15 | same |
| ViT-B/16 | frozen | 1.0 | 0 | 1 | timm augreg_in1k |
| ViT-B/16 | finetune | 1.0 | 0 | 1 | same |
| USF-MAE | frozen | 1.0 | 0 | 1 | **usf_mae_500ep.pth** |
| USF-MAE | finetune | 0.01 … 1.0 (6 fractions) | 0, 1, 2 | 18 | **usf_mae_500ep.pth** |
| **Total** | | | | **38** | **5.14 GPU-hours measured** |

Model selection always used validation macro-F1 (patient-disjoint val, 704 images, 87 patients) with patience 6. The test set was predicted once, using the selected checkpoint.

## 2. Planned vs executed

| Plan item | Planned | Executed | Consequence |
|---|---|---|---|
| USF-MAE initialisation | 100-epoch checkpoint | **500-epoch checkpoint** (nb04 edited in Colab) | 500ep is now the official model (decision 2026-11-14). `configs/usf_mae.yaml`, `configs/export.yaml`, `scripts/download_assets.sh` and `scripts/build_notebooks.py` now default to it (sha256 pinned in `python/omari_us/models/usf_mae.py`) |
| Required runs | 3 models × {frozen, finetune}, lf 1.0, s0 (6 runs, est. 4–8 GPU-h) | All 6 done | Measured times were 14–25 min per full-data run |
| Label-efficiency sweep | **frozen** regime, 3 models × 6 fractions × 3 seeds (≈54 runs) | **finetune** regime, DenseNet + USF-MAE only | No frozen sweeps. **No ViT-B/16 sweep**, so the low-label curve cannot separate ultrasound MAE pretraining from the ViT architecture |
| Seeds at 100% labels | 0–2 for every model (configs list `[0,1,2]`) | USF-MAE 3 seeds; DenseNet and ViT seed 0 only | Seed variance is known for one model only |
| DenseNet sweep | complete | Interrupted. It also re-ran `densenet161_finetune_lf1_s0` and overwrote that run's `best.pt` and `history.json` | See `docs/PROVENANCE.md` |
| Website export | `export_website_artifacts.py` → `website/public/experiments/` | nb06 wrote `ART/website_bundle/` (schema 1.0, with defects); nb05 wrote `ART/website_export/` | Superseded by `scripts/finalize_website_bundle.py` (schema 1.1) |
| `prepare_dataset.py` writes `website/public/experiments/dataset_stats.json` | yes | nb01 wrote `ART/dataset_stats.json` instead | Folded into the finalized bundle |

## 3. Caveat: the "frozen" probe was not strictly frozen

The completed frozen runs call `model.train()` every epoch with the backbone parameters frozen (`requires_grad=False`). In train mode, **DenseNet BatchNorm running statistics were still updated** on FETAL_PLANES_DB, and dropout/drop-path was active for the ViTs. So the frozen numbers (macro-F1 0.7657 / 0.8038 / 0.8039) are a "linear head + BN-statistics adaptation" probe, not a pure linear probe. This matters most for DenseNet. The code fix (backbone in `eval()` mode during frozen training) affects **future** runs only. The reported frozen numbers come from the original behaviour and were not re-run.

## 4. Reproducing or extending

```bash
pip install -r requirements.txt && pip install -e python
bash scripts/download_assets.sh data            # FETAL_PLANES_DB (md5-checked) + USF-MAE 500-ep checkpoint
python scripts/prepare_dataset.py --data-root data --out artifacts/splits
for m in densenet161 vit_b16 usf_mae; do
  python scripts/train.py --config configs/$m.yaml --regime frozen
  python scripts/train.py --config configs/$m.yaml --regime finetune
done
python scripts/train.py --config configs/usf_mae.yaml --regime finetune --all-fractions   # sweep as executed
```

To reuse the **exact split** of the reported runs, use the tracked `content/omari_artifacts/omari_artifacts/splits/split_manifest.csv`, re-rooted with `--data-root`. The label-fraction training subsets were not saved and could not be reproduced exactly. Training is seeded but not bit-deterministic.

Then build the website bundle as described in the README (`scripts/finalize_website_bundle.py`, then `scripts/check_provenance.py`).

### Optional, scientifically valuable (not run)

| Run | Why |
|---|---|
| ViT-B/16 finetune sweep (5 fractions × 3 seeds) | The controlled comparison for USF-MAE's low-label advantage (same architecture) |
| DenseNet and ViT lf1 seeds 1–2 | Equal seed counts at full labels |
| `densenet161_finetune_lf1_s0` rerun to completion in a fresh directory | Restores a checkpoint that matches its summary |
| Truly frozen probe (backbone in `eval()`) | Pure linear-probe numbers |
| USF-MAE pretrained **without** FETAL_PLANES_DB (if the authors can provide one) | Removes the transductive pretraining overlap |
| Leave-one-machine-out | Stronger domain-shift test than by-machine subgroups |
| USF-MAE with `global_pool: avg` | Checks that conclusions do not depend on the head |

## 5. Historical: original pre-training plan (2026-11-04)

> Training could not run in the build environment: no GPU, and the dataset and pretrained weights were unreachable. Required: 6 runs (3 models × frozen/finetune, ≈4–8 GPU-hours). Recommended: label-efficiency sweep in the **frozen** regime (≈54 short runs) for all three models. Then `export_website_artifacts.py` → `check_provenance.py` → `npm run build`. The USF-MAE checkpoint specified was the 100-epoch release.

This plan is superseded by sections 1–4.
