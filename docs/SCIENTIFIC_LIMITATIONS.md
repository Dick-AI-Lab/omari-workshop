# Scientific limitations

What this workshop can and cannot support, written so the presenter can state it plainly.

*Updated 2026-11-14 with the real results. Numbers are from `docs/POST_TRAINING_AUDIT_2026-11-14.md`, recomputed from the run artifacts.*

## Status of results in this repository

- **Real results exist (since 2026-11-12).** 38 runs were trained in Colab. Fine-tuned test macro-F1 at 100% labels is DenseNet-161 0.9425, ViT-B/16 0.9410 and USF-MAE 0.9416 ± 0.0023 (3 seeds). Every metric recomputes exactly from the saved predictions.
- **What the results support.** At full labels the three models are **statistically indistinguishable**: Holm-adjusted McNemar p = 1, overlapping patient-cluster CIs, and paired bootstrap CIs of the differences that include 0. **No ranking at full labels is supported.** In the low-label regime, USF-MAE is clearly ahead of DenseNet-161 only at 5% labels (macro-F1 0.888 ± 0.008 vs 0.808 ± 0.023, 3 seeds each). Even that comes with the overlap and confound caveats below.
- **Class counts** are recomputed from the CSV and agree with Burgos-Artizzu et al. (2020).
- *Pre-training wording (2026-11-04), now superseded: "No model results exist yet… Code paths were verified, outputs were not."*

## Comparison caveats from the executed runs

- **A. Unequal seeds.** USF-MAE has 3 seeds at 100% labels. DenseNet-161 and ViT-B/16 have 1 seed each. DenseNet's seed sd at 50% labels (0.0072) is larger than every full-label gap between the models.
- **B. Incomplete grid.** There is no ViT-B/16 label-efficiency sweep and no frozen-regime sweep. The low-label comparison is DenseNet vs USF-MAE only, so it confounds architecture with pretraining.
- **C. "Frozen" probe.** It was not strictly frozen. The completed frozen runs kept the network in train mode, so DenseNet BatchNorm statistics adapted, and ViT dropout/drop-path was active.
- **D. Budget at 1%.** Several 1% runs peaked in the last epochs (DenseNet best epoch 24–25 of 25), so the 1% comparison is training-budget-limited and has high variance (DenseNet sd 0.073).
- **E. Validation is small and scanner-shifted.** Validation has 704 images from 87 patients and no Voluson S10 images. Best-epoch choice is noisy, because differences of about 0.005 val F1 are within noise.
- **F. Checkpoint history.** See `docs/PROVENANCE.md`. The DenseNet headline `best.pt` was overwritten by an interrupted rerun, and its metrics come from the original run. The original bundle's MAE reconstructions came from the 100ep checkpoint, while the classifier came from 500ep.
- **G. Not bit-reproducible.** Runs are seeded, but cuDNN non-determinism, unseeded workers and AMP remain. The label-fraction subsets were not saved, and a re-implementation of the sampler gives slightly different counts.

## Design of the comparison

1. **Recipes, not factors.** DenseNet-161 (supervised CNN, ImageNet-1k), ViT-B/16 (supervised ViT, ImageNet-1k, AugReg) and USF-MAE (ImageNet MAE → ultrasound MAE, ViT-B/16) differ in architecture, pretraining data, objective and input transform. The fine-tuning recipes also differ. DenseNet uses weight decay 1e-4, 1 warm-up epoch and a backbone lr ×0.1. The ViTs use weight decay 0.05, 3 warm-up epochs and layer decay 0.75. Interpolation is bilinear for DenseNet, bicubic for ViT and nearest for USF-MAE. Differences cannot be attributed to any single factor. ViT-B/16 vs USF-MAE is the closest pair (same architecture and optimiser recipe), but even there the pretraining objective *and* data differ, and so does preprocessing. That pair shows no difference at full labels (0.9410 vs 0.9416 ± 0.0023), and it was not compared at low labels.
2. **Per-model preprocessing.** Each model keeps the normalisation and interpolation its weights expect (DenseNet: ImageNet μ/σ, bilinear; ViT: 0.5/0.5, bicubic; USF-MAE: ImageNet μ/σ, nearest). All use the same geometry (stretch to 224²) so spatial maps are comparable. Stretching distorts anatomy as a function of frame size, a potential shortcut tested in Chapter 2.
3. **Hyper-parameters.** Each model follows a standard recipe for its family (configs/*.yaml). No per-model tuning on the test set; selection on patient-disjoint validation macro-F1 only. Undertuning any one model would bias the comparison, so report the configs.

## Data

4. **USF-MAE pretraining overlap: transductive exposure, not label leakage.** FETAL_PLANES_DB is one of the 46 OpenUS-46 pretraining datasets. This rests on the USF-MAE README and the project config; we did not check it image by image. USF-MAE's encoder has therefore probably seen the test images during masked-reconstruction pretraining, **without labels**. No plane label and no test label reached the encoder or the classifier head, so this is **not supervised label leakage**. It is still **transductive exposure**: the encoder learned to reconstruct these exact images and their scanner and operator statistics. That includes the Voluson S10 domain the labelled training set almost lacks, and USF-MAE is the best model on S10 (accuracy 0.937 vs 0.922–0.923). Exposure can favour USF-MAE most at **low label budgets**, which is exactly where its advantage appears. The 500-epoch checkpoint saw the corpus for longer than the 100-epoch one. This caveat must accompany any USF-MAE result. A clean comparison needs a variant pretrained without FETAL_PLANES_DB, or an external test set.
5. **Single source.** Two hospitals in one city, one health system, GE Voluson and Aloka machines, expert labels from one clinician (with technician involvement per the paper). External validity to other vendors, populations, gestational ages or acquisition protocols is unknown.
    **Scanner shift inside the official split.** Voluson S10 has **11 train, 0 val and 1,071 test** images (20% of test). Test accuracy therefore partly measures cross-scanner generalisation, and validation-based selection never sees S10. Per-scanner accuracy on S10 is DenseNet 0.923, ViT 0.922 and USF-MAE 0.937. These single-seed, unpaired values are confounded with the pretraining overlap (item 4).
6. **Within-patient correlation.** Patients contribute several images, often consecutive frames. Image-level statistics overstate the effective sample size. We use patient-cluster bootstrap for CIs; McNemar tests remain image-level and are approximate.
7. **Near-duplicates.** dHash detection (64-bit dHash, Hamming ≤ 4) is heuristic: it gives false positives between genuinely different but similar frames and false negatives under crops. *The earlier claim that cross-split pairs "should be zero or rare" with a patient-level split was wrong.* Notebook 01 found **2,685 train/val↔test pairs**, all between different patients. Only 16 of them are exact hash matches (Hamming 0), and together they touch 320 distinct test images. There are a further 317 train↔val pairs. The detector is loose: 34% of cross-split pairs have **different labels**, so most matches are generic-looking ultrasound frames, not copies. A sensitivity check on the 241 test images with a same-label near-duplicate in train/val showed **no visible inflation**. Accuracy with vs without a near-duplicate was 0.946 vs 0.948 for DenseNet, 0.942 vs 0.947 for ViT and 0.967 vs 0.945 for USF-MAE. The last gap is small and n = 241, untested. This is a documentation and limitation issue, not evidence of leakage-driven inflation.
8. **"Other" class.** A heterogeneous catch-all. Errors involving it partly reflect label definition, not model capacity.
9. **Label noise.** Not quantified here. The paper's technician error rates (5.1–6.5% top-1) indicate a non-trivial noise floor.

## Evaluation

10. **Calibration.** ECE with equal-width bins depends on the number of bins and is unstable in sparse high-confidence bins; report NLL as well. Temperature is fitted on validation, not test. The website's temperature slider is a what-if on test data for intuition only.
11. **Prevalence simulator.** Assumes pure prior shift (P(x | y) unchanged). Real deployment shift also changes image appearance.
12. **Thresholds.** Meaningful only for one-vs-rest questions; the six-way decision is an argmax.
13. **Label efficiency.** Subsets are sampled by patient and stratified by each patient's majority class, with at least one patient per class. The 1% condition is 10 patients (≈46–63 images), mostly one patient per class, not a true 1% stratified sample. Seed variance is large, so read bands, not means. Use **macro-F1, not accuracy**, because 34% of images are "Other". Only DenseNet-161 and USF-MAE were swept, in the fine-tune regime, with 3 seeds and no formal test.
14. **Subgroups.** Machine subgroups are confounded with hospital, period, operator and case mix.

## Explainability

15. **Resolution.** DenseNet-161's final maps are 7×7; ViT token maps are 14×14. Fine anatomy cannot be resolved.
16. **ViT adaptations.** Token Grad-CAM uses the input of the last block's `norm1`, a convention rather than a derivation. Layer ids `blockN` mean the input to `blocks[N]` (0-indexed), i.e. the residual stream **after N of 12 blocks**. The gradient-weighted rollout (method id `grad_rollout`, formerly `chefer`) is a simplification inspired by Chefer et al. (2021), without LRP. It is not a faithful re-implementation of that paper.
17. **Attention is not attribution.** Attention maps (Chapter 3) and rollout are not class-specific and ignore value norms, MLPs and residual paths.
18. **Faithfulness metrics have their own assumptions.** Deletion/insertion depend on the baseline (we use heavy blur). Occlusion produces out-of-distribution inputs. Randomisation tests show dependence on weights, not correctness. The executed faithfulness run used **n = 30** test images, 10 steps and no significance test. Standard deviations were 0.2–0.29. Attention rollout scored about the same as the random-map baseline, and USF-MAE `vit_gradcam` scored worse than random on both deletion and insertion.
19. **Normalised display.** Min–max scaling hides absolute magnitude; raw ranges are stored in the bundle.
20. **MetaCAM++.** Not implemented, not evaluated, and no claims are made. The interface exists for future integration.

## Toy models and simulations

21. The toy CAM model, toy attention, leakage 1-NN, hypothetical classifier and synthetic calibration data are **illustrative**. They show mechanisms; they do not estimate any property of ultrasound models. Each is badged as such.
22. The synthetic phantom is a geometric object with simulated speckle, not anatomy and not a dataset sample.

## Clinical use

23. Nothing in this workshop supports clinical deployment. See the Chapter 7 checklist for the evidence that would be required.
