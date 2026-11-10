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

## Where each provenance type appears

| Badge | Meaning | Examples |
|---|---|---|
| Live simulation | Exact maths or image processing in the browser | Preprocessing lab, receptive field, parameter counts, sin-cos similarity, masking, prevalence arithmetic |
| Model output | Precomputed by real checkpoints through the export pipeline | Predictions, feature maps, attention, attributions, reconstructions, evaluation, curves |
| Live inference | Local Python server running the checkpoints | Preprocessing lab at unexported settings |
| Illustrative | Toy or hypothetical | Toy attention, toy CAM, shortcut and sanity-check toys, hypothetical classifier, synthetic calibration, leakage 1-NN |
| Awaiting artifact | Real output required but not exported | Every model-output figure before training |
