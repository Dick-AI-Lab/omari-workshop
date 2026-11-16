# Workshop storyboard

One question runs through every chapter:

> **How do different approaches to learning image representations affect prediction, generalisation and explainability in prenatal ultrasound?**

```
medical images → data preparation → representation learning → training → evaluation → explainability → trustworthiness
   Ch 1             Ch 2                 Ch 3                     Ch 4       Ch 4          Ch 5              Ch 6–7
```

| # | Time | Chapter | Scientific beat | Main interactive figures | Data shown | Ends with |
|---|---|---|---|---|---|---|
| 1 | 0–10 | Can AI recognise this plane? | Correctness ≠ confidence ≠ reasoning | Opening challenge (guess → reveal 3 models), disagreement carousel | Exported test images + real predictions; empty state otherwise | "If all three are right, did they learn the right features?" |
| 2 | 10–25 | From pixels to training data | The tensor is a modelling decision; leakage can fake generalisation | Preprocessing lab (speckle, resolution, geometry, normalisation, histogram of the tensor), class balance with loss weights, leakage simulator | Live on any image; perturbation predictions when exported or via the live server; literature or CSV class counts | "What do models do with this tensor?" |
| 3A | 25–34 | DenseNet-161 | Locality, hierarchy, feature reuse | Kernel explorer, exact receptive-field table, dense-block connectivity, real feature maps | Live maths + exported activations | GAP + linear head (sets up CAM) |
| 3B | 34–43 | ViT-B/16 | Global mixing; positions are learned; attention ≠ explanation | Patch → token explorer with MAE sin-cos similarity, attention explorer (toy ↔ model, block/head/query, depth strip) | Live + exported attention | "Attention maps are seductive…" |
| 3C | 43–50 | USF-MAE | Self-supervision on ultrasound changes what the encoder already knows | Masking explorer (ratio, seed, compute), authentic reconstructions, pretraining vs fine-tuning schematic | Live masking; exported reconstructions only | Overlap callout + "only 'it depends' is defensible" |
| 4 | 50–70 | Training, evaluation, generalisation | Measure, don't assume: macro-F1, CIs, prevalence, calibration, label efficiency, domain shift | Confusion explorer with image drill-down, per-class bars, ROC/threshold, prevalence simulator, reliability + temperature, learning curves, by-machine table | All from notebook 05 exports; prevalence and calibration fall back to a labelled hypothetical classifier | Two caveats travel with every result |
| — | 70–80 | Break | — | Self-guided prompts | — | — |
| 5 | 80–105 | From CAM to MetaCAM++ | Explanations are models of models, with assumptions | Toy CAM / Grad-CAM / Grad-CAM++ with deletion curve; real attribution lab (model × method × class × layer, opacity, threshold, colour scale, correct vs incorrect); compatibility table; MetaCAM++ slot | Toy (illustrative) + exported maps | Attention routing ≠ class attribution |
| 6 | 105–115 | Model detective | Plausible ≠ faithful; test hypotheses by intervention | Hypothesis lock → perturbation evidence table (Δp, flips, map ρ), live overlay detector, shortcut toy, randomisation sanity check | Exported controlled experiments + live/toy | Occlusion question |
| 7 | 115–120 | What would you trust? | Evidence needed before deployment | Opening image again, deployment checklist, takeaways, notebooks | — | Questions |

## What the real results let each chapter say (added 2026-11-14)

The verified numbers are in `docs/POST_TRAINING_AUDIT_2026-11-14.md`. The data shown comes from the finalized bundle (`scripts/finalize_website_bundle.py`, schema 1.1).

| Chapter | Supported message | Do not say |
|---|---|---|
| 1 | Three models agree on most images; disagreements exist and are instructive | That any model "sees" the anatomy |
| 2 | The split is patient-disjoint (overlap 0). A loose hash detector still finds 2,685 look-alike train/val↔test pairs (different patients), with no visible accuracy inflation. Voluson S10 is almost absent from training (11 images) but makes up 20% of test | "There are no near-duplicates" |
| 3C | USF-MAE (500-epoch checkpoint) reconstructs masked ultrasound. Its pretraining corpus included FETAL_PLANES_DB images without labels: transductive exposure, not label leakage | That reconstructions show what the classifier uses |
| 4 | Full labels: macro-F1 0.9425 / 0.9410 / 0.9416 ± 0.0023, **statistically indistinguishable**. Low labels (macro-F1): USF-MAE ahead of DenseNet clearly only at 5% (0.888 ± 0.008 vs 0.808 ± 0.023). ViT low-label was **not run** and is shown as such. All models are under-confident; temperature scaling fixes ECE | Any full-label ranking; "foundation models are better" |
| 5 | Methods disagree; ViT layer names read "after block N of 12"; gradient-weighted rollout is a simplified Chefer-inspired variant | That heatmaps are ground truth |
| 6 | Faithfulness (n = 30): attention rollout ≈ random; USF-MAE token Grad-CAM worse than random | Significance claims |

## Where each provenance type appears

| Badge | Meaning | Examples |
|---|---|---|
| Live simulation | Exact maths or image processing in the browser | Preprocessing lab, receptive field, parameter counts, sin-cos similarity, masking, prevalence arithmetic |
| Model output | Precomputed by real checkpoints through the export pipeline | Predictions, feature maps, attention, attributions, reconstructions, evaluation, curves |
| Live inference | Local Python server running the checkpoints | Preprocessing lab at unexported settings |
| Illustrative | Toy or hypothetical | Toy attention, toy CAM, shortcut and sanity-check toys, hypothetical classifier, synthetic calibration, leakage 1-NN |
| Awaiting artifact | Real output required but not exported (or a run that was never done, shown as "not run") | Every model-output figure before the bundle is installed; ViT-B/16 label efficiency below 100% |
