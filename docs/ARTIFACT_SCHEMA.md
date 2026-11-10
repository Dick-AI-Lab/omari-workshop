# Artifact bundle schema v1

The contract between `python/omari_us/export/website_export.py` (writer) and `website/src/data/types.ts` (reader). Everything lives in `website/public/experiments/`. All paths inside JSON are relative to that folder. `scripts/check_provenance.py` validates a bundle.

```
experiments/
  manifest.json              Manifest (required to enable any model-output figure)
  dataset_stats.json         DatasetStats (can exist alone, before any model is trained)
  evaluation.json            Evaluation
  training_curves.json       TrainingCurves
  label_efficiency.json      LabelEfficiency
  samples/<Image_name>/
    image.png                8-bit grey, longest side ≤ max_side; sample predictions are computed on exactly these pixels
    features/<layer>_<c>.png DenseNet-161 channel maps (min–max scaled)
    attention/<model>_cls.json        {blocks, heads, grid, cls[block][head][196], rollout[196], provenance}
    attention/<model>_block<b>.u16.bin little-endian uint16, heads×197×197, probability × 65535
    attributions/<model>_<method>_<layer>_c<k>.png   8-bit map at native resolution (7×7, 14×14 …)
    reconstructions/r<pct>_s<seed>_{full,pasted}.png
    perturbations/<type>_<param>.png and per-model attribution PNGs
  misclassified/<Image_name>.png     thumbnails for the confusion-matrix drill-down
  export_log.txt
```

## manifest.json

| Field | Type | Notes |
|---|---|---|
| `schema_version` | `"1.0"` | the website accepts `1.x` |
| `bundle_id`, `generated_at`, `generator{script, version, git_commit, command}` | | |
| `dataset{name, license, source_url, citation, split_protocol}` | | |
| `classes` | `string[6]` | index order used everywhere |
| `models[]` | `{id, display_name, checkpoint_id, checkpoint_sha256, initialization, finetuning, params, input{size, resize, interpolation, mean, std, channels}, notes}` | `id ∈ densenet161, vit_b16, usf_mae` |
| `samples[]` | SampleInfo | below |
| `dataset_stats`, `evaluation`, `training_curves`, `label_efficiency` | path | optional |

### SampleInfo

`{id, image, width, height, label, brain_plane?, machine?, operator?, split, patient, tags[], predictions{model: {probs[6], pred}}, features?, attention?, attributions?, reconstructions?, perturbations?}`

- `tags` include `opening`, `disagreement`, `class-<k>`, `misclassified-<model>`.
- `attributions[]`: `{model, method, layer, target_class, png, raw_min, raw_max, resolution[2], provenance}`. `method ∈ cam, gradcam, gradcampp, vit_gradcam, vit_gradcampp, attention_rollout, chefer, metacampp`.
- `reconstructions[]`: `{mask_ratio, seed, masked[], reconstruction_png, pasted_png, loss_masked_mse, provenance}`. The website snaps its slider to exported ratios and never synthesises a reconstruction.
- `perturbations[]`: `{type, param, label, image_png, predictions{model}, attributions{model: {method, png}}, provenance}`. Semantics are defined in `omari_us/datasets/perturbations.py` and mirror `website/src/lib/imageOps.ts` (brightness additive, contrast about 0.5, speckle Gamma(L,1/L) with L = 1, resolution = area-down + nearest-up).

## Provenance object (every model-derived item)

`{script, generated_at, git_commit?, checkpoint_id?, checkpoint_sha256?, params?, note?}`

## evaluation.json

`{provenance, split, n, per_model{model: {confusion[6][6], accuracy, macro_f1, balanced_accuracy, per_class[{precision, recall, f1, support}], ci95{accuracy, macro_f1}, ece, nll, by_machine?, temperature_fit_on_val?, ece_after_temperature?}}, comparisons[{a, b, test, statistic, p_value, note}], test_predictions?{ids, labels, machines?, probs{model: number[n][6]}}, misclassified[{id, image, label, preds{model}}]}`

CIs are patient-cluster percentile bootstrap. McNemar tests are exact, Holm-adjusted in `note`.
