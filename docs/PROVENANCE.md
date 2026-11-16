# Provenance of website artifacts

*Created 2026-11-14.* For every artifact the website displays, this document records its source run or checkpoint, the hash, and the generator. `ART/` = `content/omari_artifacts/omari_artifacts/` (a local copy of the Colab Drive folder `/content/drive/MyDrive/omari_artifacts`). The website reads only `website/public/experiments/`. That folder is produced by `scripts/finalize_website_bundle.py` from `ART/` and is not committed (see its README).

## 1. Checkpoints

| Checkpoint | sha256 | Role | Generator |
|---|---|---|---|
| `ART/checkpoints/usf_mae_500ep.pth` | `c659a06170b0f6967ae7f6d066681b8dea2c122a5f325bb8a82df11ffb3a2533` | **Official USF-MAE pretrained model.** It initialised all 19 USF-MAE runs (summaries record `sha256:c659a06170b0`) and is used for 500ep reconstructions if `--reconstruct` was run | USF-MAE authors (Google Drive `15n71qHXSBjEjrLtCOzvBb6gTsGhh7NUg`) |
| `ART/checkpoints/usf_mae_100ep.pth` | `f815c629878c17136985af9f4fdc81c2cfa02a94e4d992c026699957f75ccb66` | **Historical.** Used only for the reconstructions in the original Colab bundle (now `reconstructions_legacy`). Not used by any classifier | USF-MAE authors (Drive `1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr`) |
| `ART/runs/densenet161_finetune_lf1_s0/best.pt` | file `456847049ab204aac38fd4f7a189965d3069a534577eee056fb9a7bdfb937ff2`; summary records `20d40d72c51bf892a499edfc835eb3a14af9e3e9dd485af46006df4029fa72f8` | DenseNet-161 headline model (XAI, features, perturbations) | nb02. **See section 3** |
| `ART/runs/vit_b16_finetune_lf1_s0/best.pt` | `49fe4168a8922632d88aa36fa1e85c55dc14db32d4b9159549c99fadbe684d8c` (matches summary) | ViT-B/16 headline model | nb03 |
| `ART/runs/usf_mae_finetune_lf1_s0/best.pt` | `65680650b75f30ff7f38d88afa27c820de87a20ebc3e7c42d761a9b5df9c29d1` (matches summary) | USF-MAE headline model (seed 0) | nb04, from 500ep |
| `ART/runs/usf_mae_finetune_lf1_s1/best.pt`, `…_s2/best.pt` | `56be95e53058635fa768e47ad4ac6fdb162769f562fd919529488bba03eb638e`, `8b0fd24e1ec3a06f42e8d3d539d6c79238500a309704a267436499a81c8c223a` | Seed summary only | nb04, from 500ep |

Each of the other 32 runs also records its `best.pt` sha256 in `summary.json`, and each matches the file on disk.

## 2. Bundle files (`website/public/experiments/`, schema 1.1)

| File | Source in ART | Source run(s) / checkpoint | Generator |
|---|---|---|---|
| `manifest.json` | `website_bundle/manifest.json` (nb06, 2026-11-12T01:29Z, schema 1.0) | Headline `best.pt` of the three models (hashes above) | Rewritten by `finalize_website_bundle.py`: schema 1.1, `models[].notes` (overlap disclosure), `models[].pretraining`, `provenance_notes`, `sample_predictions_source`, `grad_rollout` rename, `layer_label` / `method_label` |
| `samples[].predictions` | `runs/<headline>/test_predictions.npz` | Same evaluation that produced the metrics (original-resolution images) | finalize script. The 1.0 values, recomputed on 640 px PNGs, disagreed with `evaluation.json` for 2/72 USF-MAE cases |
| `samples/<id>/image.png`, `features/`, `attention/`, `attributions/`, `perturbations/` | `website_bundle/samples/` | Headline checkpoints. **DenseNet maps come from the rerun `best.pt` (`456847…`)**, see section 3 | `omari_us.export.website_export` in nb06; copied unchanged (PNG filenames keep `_chefer_`) |
| `samples/<id>/reconstructions_500ep/` → `reconstructions[]` | generated locally if `--reconstruct` was run | `usf_mae_500ep.pth` (`c659a061…`) | `finalize_website_bundle.py --reconstruct` |
| `samples/<id>/reconstructions/` → `reconstructions_legacy[]` (or `reconstructions[]` if 500ep was not generated) | `website_bundle/samples/*/reconstructions/` | `usf_mae_100ep.pth` (`f815c629…`), original provenance kept | nb06 |
| `evaluation.json` | `website_bundle/evaluation.json` (numbers identical to nb05's `website_export/evaluation.json`) | `test_predictions.npz` of the three headline runs; DenseNet from the **original** run (summary `20d40d…`) | nb06 exporter; `seed_summary` added by finalize script from `usf_mae_finetune_lf1_s{0,1,2}/summary.json` (sd ddof = 1) |
| `label_efficiency.json` | rebuilt from `runs/*_finetune_lf*_s*/summary.json` | 15 DenseNet + 18 USF-MAE sweep runs, plus the lf1 runs; nulls where not run (ViT < 100%, DenseNet/ViT lf1 s1–s2) | finalize script. The 1.0 bundle file held a single frozen point (config bug), and `website_export/` held `NaN` |
| `training_curves.json` | `website_bundle/training_curves.json` | `history.json` of the finetune lf1 s0 runs. **The DenseNet curve is the truncated rerun history (10 epochs)**, not the original run's | nb06 exporter; see the finalize script's `provenance_notes` for how this is labelled |
| `dataset_stats.json` | `ART/dataset_stats.json` (nb01) + `splits/split_manifest.csv` + `splits/near_duplicates.csv` | — | finalize script (adds split, near-duplicate and machine-by-split fields that the 1.0 bundle dropped) |
| `faithfulness.json` | `ART/faithfulness.csv` (nb06 cell 10) | Headline checkpoints (DenseNet: rerun `best.pt`); 30 random test images (`random_state=42`), 10 steps, Gaussian-blur baseline, true-label probability | finalize script (parsed; the CSV has no provenance of its own) |
| `misclassified/*.png` | `website_bundle/misclassified/` | dataset images (CC BY 4.0) | nb06 exporter |
| `export_log.txt` | `website_bundle/export_log.txt` | — | nb06 |

Files in `ART/` that the website does **not** use: `website_export/` (superseded nb05 export, contains `NaN`), `usf_mae_test_embeddings.npz` (nb04 PCA plot only), the prediction files of non-headline runs (used only through their `summary.json`), and all frozen-run checkpoints.

## 3. DenseNet-161 headline checkpoint status

**What happened.** `densenet161_finetune_lf1_s0` finished at 2026-11-11T00:15:51Z with `best.pt` sha256 `20d40d72…` (best epoch 7, val macro-F1 0.9589369951650811). At about 03:00Z, the interrupted label-fraction sweep re-ran this configuration in the same directory. The rerun was killed after epoch 10, but it had already **overwritten `best.pt`** (now `456847…`, created 03:14:12Z, epoch 7, same val macro-F1 to every digit) **and `history.json`** (truncated at epoch 10). `summary.json`, `val_predictions.npz` and `test_predictions.npz` are from the original run.

**Consequences.**
- All DenseNet metrics (accuracy 0.9474, macro-F1 0.9425, CIs, McNemar) come from the original run's predictions.
- The DenseNet XAI maps, feature maps, perturbation predictions and faithfulness scores were computed with the rerun `best.pt`.
- The DenseNet training curve is the rerun's truncated history.

**Functional-equivalence check (2026-11-14).** The rerun `best.pt` was re-evaluated locally (RTX 4080 SUPER, torch 2.6.0+cu124) on the exact val/test images of the split manifest. It used the run's own eval preprocessing and was compared with the original run's saved predictions by image id. The ViT-B/16 headline checkpoint, whose hash matches its summary, was run the same way as a control for hardware and software numerical drift. Source: `densenet_equivalence.json` (coordinator scratch analysis).

| | DenseNet rerun `best.pt` vs original predictions | ViT-B/16 control |
|---|---|---|
| Test, AMP on (as in the original runs) | argmax agreement **5,271 / 5,271**; max abs Δp 0.0016 (mean 2.8e-5); recomputed accuracy 0.9474 / macro-F1 0.9425 (identical) | 5,271 / 5,271; max abs Δp 0.0056; identical metrics |
| Test, AMP off | 5,270 / 5,271 (1 flip); max abs Δp 0.054 (mean 4.4e-4); acc 0.9476 / macro-F1 0.9428 | 5,270 / 5,271 (1 flip); max abs Δp 0.0051; acc 0.9467 / macro-F1 0.9412 |
| Val, AMP on | 704 / 704; max abs Δp 0.0016; macro-F1 0.9589 (identical) | 704 / 704; max abs Δp 0.0022; identical |
| Val, AMP off | 704 / 704; max abs Δp 0.023 | 704 / 704; max abs Δp 0.0038 |

**Interpretation.** Under the original inference settings (fp16 autocast), the rerun checkpoint reproduces every original test and validation prediction. Its probability differences are no larger than those of the ViT control, which has a verified hash match. We therefore treat the rerun weights as **functionally equivalent** to the original for the purposes of the website. The DenseNet maps and the DenseNet metrics can be read together. Two things remain true:
- The weights are **not byte-identical** to the hash in `summary.json`.
- With AMP off, the DenseNet probabilities drift about 10× more than the control's. DenseNet's fp32 outputs are more sensitive than ViT's, but this has no effect on the reported metrics, which were produced with AMP.

The bundle records this in `manifest.provenance_notes`. A clean fix would be to re-run the configuration to completion in a fresh directory (not done; no retraining in this phase).

## 4. Reconstruction checkpoint history

- **2026-11-12 (nb06):** reconstructions for all 24 samples (4 mask ratios × seed 0) were generated with `usf_mae_100ep.pth`, while the USF-MAE classifier had been initialised from `usf_mae_500ep.pth`. The original bundle therefore mixed two pretraining runs.
- **2026-11-14:** 500ep was adopted as official. `finalize_website_bundle.py --reconstruct` regenerates reconstructions from `usf_mae_500ep.pth` into `reconstructions_500ep/` and makes them primary. The 100ep images are kept as `reconstructions_legacy` with their original provenance. If the bundle was built without `--reconstruct`, `reconstructions` remain the 100ep ones, honestly labelled. Check `samples[].reconstructions[].provenance.checkpoint_id` in the installed manifest.

## 5. Code version

`git_commit` is `null` in every `summary.json` and in the original manifest, because notebooks ran `git` outside the cloned repository. The runs used fresh clones of the Dick-AI-Lab repository plus the notebook edits now committed as the executed notebooks (`notebooks/*.ipynb`). The code was modified after commit `3ad7b36`. The run configuration is fully recorded in each `summary.json` (`config`, `input`, `init_source`).
