# Scientific limitations

What this workshop can and cannot support, written so the presenter can state it plainly.

## Status of results in this repository

- **No model results exist yet.** Nothing was trained on FETAL_PLANES_DB in the build environment (no GPU; dataset and weights unreachable). Every model-output figure shows an explicit empty state until the export pipeline is run on real checkpoints.
- **Code paths were verified, outputs were not.** The full pipeline (training → export → website) was executed end to end on synthetic images with random weights. That proves the code runs and the schema is consistent; it says nothing about performance.
- **Literature numbers are labelled as such.** Class counts come from Burgos-Artizzu et al. (2020), transcribed, until notebook 01 recomputes them from the CSV.

## Design of the comparison

1. **Recipes, not factors.** DenseNet-161 (supervised CNN, ImageNet-1k), ViT-B/16 (supervised ViT, ImageNet-1k, AugReg) and USF-MAE (ImageNet MAE → ultrasound MAE, ViT-B/16) differ in architecture, pretraining data, objective, augmentation and input transform. Differences cannot be attributed to any single factor. ViT-B/16 vs USF-MAE is the closest pair (same architecture), but even there pretraining objective *and* data differ.
2. **Per-model preprocessing.** Each model keeps the normalisation and interpolation its weights expect (DenseNet: ImageNet μ/σ, bilinear; ViT: 0.5/0.5, bicubic; USF-MAE: ImageNet μ/σ, nearest). All use the same geometry (stretch to 224²) so spatial maps are comparable. Stretching distorts anatomy as a function of frame size, a potential shortcut tested in Chapter 2.
3. **Hyper-parameters.** Each model follows a standard recipe for its family (configs/*.yaml). No per-model tuning on the test set; selection on patient-disjoint validation macro-F1 only. Undertuning any one model would bias the comparison, so report the configs.

## Data

4. **USF-MAE pretraining overlap.** FETAL_PLANES_DB is one of the 46 OpenUS-46 pretraining datasets. USF-MAE's encoder has seen these images (without labels), possibly including our test split. This is a transductive advantage over ImageNet-initialised models. It must accompany any USF-MAE result. A clean comparison needs a variant pretrained without FETAL_PLANES_DB or an external test set.
5. **Single source.** Two hospitals in one city, one health system, GE Voluson and Aloka machines, expert labels from one clinician (with technician involvement per the paper). External validity to other vendors, populations, gestational ages or acquisition protocols is unknown.
6. **Within-patient correlation.** Patients contribute several images, often consecutive frames. Image-level statistics overstate the effective sample size. We use patient-cluster bootstrap for CIs; McNemar tests remain image-level and are approximate.
7. **Near-duplicates.** dHash detection is heuristic (false positives between genuinely different but similar frames; false negatives under crops). Notebook 01 reports cross-split pairs; with the patient-level split these should be zero or rare.
8. **"Other" class.** A heterogeneous catch-all. Errors involving it partly reflect label definition, not model capacity.
9. **Label noise.** Not quantified here. The paper's technician error rates (5.1–6.5% top-1) indicate a non-trivial noise floor.

## Evaluation

10. **Calibration.** ECE with equal-width bins depends on the number of bins and is unstable in sparse high-confidence bins; report NLL as well. Temperature is fitted on validation, not test. The website's temperature slider is a what-if on test data for intuition only.
11. **Prevalence simulator.** Assumes pure prior shift (P(x | y) unchanged). Real deployment shift also changes image appearance.
12. **Thresholds.** Meaningful only for one-vs-rest questions; the six-way decision is an argmax.
13. **Label efficiency.** Subsets are sampled by patient and stratified by each patient's majority class. Small fractions (1%) contain few patients per class, so seed variance is large; read bands, not means.
14. **Subgroups.** Machine subgroups are confounded with hospital, period, operator and case mix.

## Explainability

15. **Resolution.** DenseNet-161's final maps are 7×7; ViT token maps are 14×14. Fine anatomy cannot be resolved.
16. **ViT adaptations.** Token Grad-CAM uses the input of the last block's `norm1`, a convention rather than a derivation. The gradient-weighted rollout is a simplification of Chefer et al. (2021) without LRP; it is not a faithful re-implementation of that paper.
17. **Attention is not attribution.** Attention maps (Chapter 3) and rollout are not class-specific and ignore value norms, MLPs and residual paths.
18. **Faithfulness metrics have their own assumptions.** Deletion/insertion depend on the baseline (we use heavy blur); occlusion produces out-of-distribution inputs; randomisation tests show dependence on weights, not correctness.
19. **Normalised display.** Min–max scaling hides absolute magnitude; raw ranges are stored in the bundle.
20. **MetaCAM++.** Not implemented, not evaluated, and no claims are made. The interface exists for future integration.

## Toy models and simulations

21. The toy CAM model, toy attention, leakage 1-NN, hypothetical classifier and synthetic calibration data are **illustrative**. They show mechanisms; they do not estimate any property of ultrasound models. Each is badged as such.
22. The synthetic phantom is a geometric object with simulated speckle, not anatomy and not a dataset sample.

## Clinical use

23. Nothing in this workshop supports clinical deployment. See the Chapter 7 checklist for the evidence that would be required.
