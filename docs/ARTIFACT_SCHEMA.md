# Artifact bundle schema v1.1

The contract between the bundle writers and the website reader (`website/src/data/types.ts`).

- `python/omari_us/export/website_export.py` writes schema 1.0 (it produced the original Colab bundle).
- `scripts/finalize_website_bundle.py` writes schema 1.1 from the Colab outputs.

Everything lives in `website/public/experiments/`. All paths inside JSON are relative to that folder. `scripts/check_provenance.py` validates a bundle. **The website accepts both `1.0` and `1.1`.** Every 1.1 addition is optional for the reader, which falls back gracefully when a field is absent.

*Changed 2026-11-14 (1.0 → 1.1):*
- method id `chefer` → `grad_rollout`
- `method_label` / `layer_label` on attributions
- `models[].notes` (non-null) and `models[].pretraining`
- `provenance_notes`, `sample_predictions_source`
- 500ep `reconstructions` + `reconstructions_legacy`
- a `faithfulness.json` pointer
- a seed summary in `evaluation.json`
- a full finetune grid with nulls in `label_efficiency.json`
- split, near-duplicate and machine-by-split fields in `dataset_stats.json`
- **No `NaN` anywhere**: files are written with `allow_nan=False`, and missing values are `null`

```
experiments/
  manifest.json              Manifest (required to enable any model-output figure)
  dataset_stats.json         DatasetStats (can exist alone, before any model is trained)
  evaluation.json            Evaluation
  training_curves.json       TrainingCurves
  label_efficiency.json      LabelEfficiency
  faithfulness.json          Faithfulness (1.1)
  samples/<Image_name>/
    image.png                8-bit grey, longest side ≤ max_side (640)
    features/<layer>_<c>.png DenseNet-161 channel maps (min–max scaled)
    attention/<model>_cls.json        {blocks, heads, grid, cls[block][head][196], rollout[196], provenance}
    attention/<model>_block<b>.u16.bin little-endian uint16, heads×197×197, probability × 65535
    attributions/<model>_<method>_<layer>_c<k>.png   8-bit map at native resolution (7×7, 14×14 …)
    reconstructions/r<pct>_s<seed>_{full,pasted}.png        legacy (100ep) reconstructions
    reconstructions_500ep/r<pct>_s<seed>_{full,pasted}.png  official (500ep) reconstructions, if generated
    perturbations/<type>_<param>.png and per-model attribution PNGs
  misclassified/<Image_name>.png     thumbnails for the confusion-matrix drill-down
  export_log.txt
```

Attribution PNG filenames for the gradient-weighted rollout keep the historical `_chefer_` infix. Only the manifest `method` value changes.

## manifest.json

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `"1.0"` \| `"1.1"` | the website accepts `1.x` |
| `bundle_id`, `generated_at`, `generator{script, version, git_commit, command}` | | `git_commit` is `null` in the Colab bundle (git ran outside the repo) |
| `dataset{name, license, source_url, citation, split_protocol}` | | |
| `classes` | `string[6]` | index order used everywhere: abdomen, brain, femur, thorax, cervix, other |
| `models[]` | `{id, display_name, checkpoint_id, checkpoint_sha256, initialization, finetuning, params, input{size, resize, interpolation, mean, std, channels}, notes, pretraining?}` | `id ∈ densenet161, vit_b16, usf_mae` |
| `models[].notes` | `string` (1.1: non-null) | `usf_mae` **must** disclose: FETAL_PLANES_DB images (unlabelled) were in the USF-MAE self-supervised pretraining corpus (OpenUS-46). This is not supervised label leakage but transductive exposure that may favour USF-MAE, especially at low label fractions |
| `models[].pretraining` (1.1, optional) | `{checkpoint, sha256, epochs, objective, corpus, overlap_with_test}` | usf_mae: `usf_mae_500ep.pth`, `c659a061…b3a2533`, 500, "masked autoencoding", "OpenUS-46 (unlabelled ultrasound; includes FETAL_PLANES_DB)". densenet161: ImageNet-1k supervised. vit_b16: ImageNet-1k augreg supervised |
| `provenance_notes` (1.1) | `string[]` | Global caveats: DenseNet checkpoint status, reconstruction checkpoint, per-sample prediction source |
| `sample_predictions_source` (1.1) | `string` | `samples[].predictions` come from `runs/<headline run>/test_predictions.npz` (the same evaluation that produced the metrics, original-resolution images), **not** recomputed on the downscaled PNGs |
| `samples[]` | SampleInfo | below |
| `dataset_stats`, `evaluation`, `training_curves`, `label_efficiency`, `faithfulness` (1.1) | path | optional |

### SampleInfo

`{id, image, width, height, label, brain_plane?, machine?, operator?, split, patient, tags[], predictions{model: {probs[6], pred}}, features?, attention?, attributions?, reconstructions?, reconstructions_legacy?, perturbations?}`

- `tags` include `opening`, `disagreement`, `class-<k>`, `misclassified-<model>`.
- `features[]`: `{layer, shape, channels[{index, png, min, max, mean}], selection}`. `attention`: `{cls, full[{block, path, heads, tokens}]}`.
- `attributions[]`: `{model, method, method_label (1.1), layer, layer_label (1.1), target_class, png, raw_min, raw_max, resolution[2], provenance}`.
  - `method ∈ cam, gradcam, gradcampp, vit_gradcam, vit_gradcampp, attention_rollout, grad_rollout, metacampp`. In 1.0 bundles `grad_rollout` appears as `chefer`, and readers should map it.
  - `grad_rollout` is displayed as "Gradient-weighted attention rollout". It is a simplified variant inspired by Chefer et al. (2021), without LRP.
- `reconstructions[]`: `{mask_ratio, seed, masked[], reconstruction_png, pasted_png, loss_masked_mse, provenance}`.
  - In 1.1 these are the primary **500ep** reconstructions (`provenance.checkpoint_id = "usf_mae_500ep.pth"` with its sha256) when they were generated.
  - The original 100ep entries move unchanged to `reconstructions_legacy[]`, keeping their original provenance (`usf_mae_100ep.pth`, `f815c629…`).
  - If 500ep generation was not possible, `reconstructions` stays 100ep with honest provenance.
  - The website snaps its slider to exported ratios and never synthesises a reconstruction.
- `perturbations[]`: `{type, param, label, image_png, predictions{model}, attributions{model: {method, png}}, provenance}`. Semantics are defined in `omari_us/datasets/perturbations.py` and mirror `website/src/lib/imageOps.ts` (brightness additive, contrast about 0.5, speckle Gamma(L,1/L) with L = 1, resolution = area-down + nearest-up).

### Layer naming convention (ViT / USF-MAE)

- Layer ids `block5`, `block8` and `block11` mean "the input to `model.blocks[N]` (0-indexed)", which is the residual stream **after N transformer blocks**.
- Display convention everywhere is 1-indexed out of 12 blocks: `blockN` is shown as "after block N of 12".
- In attention JSON, `cls[i]` (i = 0..11) is displayed as "Block i+1".
- The 1.1 exporter writes the display string to `layer_label`.
- DenseNet layers are `features.denseblock{1..4}` and `act` (the final ReLU(norm5) features, 2208×7×7).

## Provenance object (every model-derived item)

`{script, generated_at, git_commit?, checkpoint_id?, checkpoint_sha256?, params?, note?}`

## evaluation.json

`{provenance, split, n, per_model{model: {confusion[6][6], accuracy, macro_f1, balanced_accuracy, per_class[{precision, recall, f1, support}], ci95{accuracy, macro_f1}, ece, nll, by_machine?, temperature?, ece_after_temperature?}}, comparisons[{a, b, test, statistic, p_value, note}], test_predictions?{ids, labels, machines?, probs{model: number[n][6]}}, misclassified[{id, image, label, preds{model}}], seed_summary? (1.1)}`

- CIs are a patient-cluster percentile bootstrap (1,000 resamples, seed 0). McNemar tests are exact and image-level, with Holm adjustment noted in `note`.
- The temperature fields keep the names used by the source exporter, and the finalize script passes them through. The temperature is fitted on validation.
- `seed_summary` (1.1): `{usf_mae: {seeds: [0,1,2], runs: [...], accuracy: [...], macro_f1: [...], mean: {accuracy, macro_f1}, sd: {accuracy, macro_f1}}}`, from the three `usf_mae_finetune_lf1_s*` summaries. **sd is the sample sd (ddof = 1).**

## label_efficiency.json (1.1)

```
{ provenance, regime: "finetune", fractions: [0.01,0.05,0.1,0.25,0.5,1.0], seeds: [0,1,2],
  per_model: { <mid>: { macro_f1: (number|null)[][], accuracy: (number|null)[][], runs: (string|null)[][] } },  // fractions × seeds
  availability: { <mid>: string },   // e.g. vit_b16: "Only 100% labels, seed 0, was run; no low-label sweep."
  notes: string[] }
```

All three models are present, with `null` where a run does not exist. The 1.0 bundle had `regime: "frozen"` with a single point because of a config bug.

## dataset_stats.json

The 1.0 fields are kept. In 1.0, `train_images` (7,129) means train + val (the authors' official training set). 1.1 adds:

- `class_counts: {train, val, test, train_plus_val}`, where `train` **excludes** val.
- `split: {train_images, val_images, test_images, train_patients, val_patients, test_patients, patient_overlap: 0, protocol}`.
- `near_duplicates: {method, threshold, pairs, groups, images_in_groups, cross_split_pairs, cross_split_by_pair{train_val, train_test, val_test}, test_images_involved, label_discordant_fraction_cross_split, note}`. `cross_split_pairs` counts only (train or val)↔test pairs (2,685), and the 317 train↔val pairs are in `cross_split_by_pair.train_val`.
- `machines_by_split?: {train, val, test}`. Voluson S10 has 11 / 0 / 1,071.

## faithfulness.json (1.1)

`{provenance, n_images: 30, steps: 10, baseline: "blurred image", target: "true-label probability", rows: [{model, method, layer, deletion_auc_mean, deletion_auc_sd, insertion_auc_mean, insertion_auc_sd}], random_baseline?, notes}`

This is parsed from `ART/faithfulness.csv` (notebook 06). Lower deletion AUC and higher insertion AUC are better. The CSV itself carries no provenance, so n, steps and baseline come from the notebook code and outputs.
