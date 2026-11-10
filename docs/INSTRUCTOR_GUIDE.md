# Instructor guide: minute by minute

**From Pixels to Predictions: Developing AI Models for Medical Image Interpretation**
OMARI Seminar Series · 120 minutes including a 10-minute break · mixed audience (clinicians, students, researchers, AI scientists)

## Before the day

| When | Task | Check |
|---|---|---|
| 1 week before | Train the models and export the bundle (see `docs/TRAINING_PLAN.md`) | `python scripts/check_provenance.py website/public/experiments` passes |
| 1 week before | Choose the opening image: a test image all three models classify correctly with different confidence, or one where they disagree. Set `samples.opening` in `configs/export.yaml` and re-export | Chapter 1 shows it |
| 2 days before | `cd website && npm run build`. Open `dist/` on the presentation laptop (works offline) | No "Awaiting artifact" badges in Chapters 1, 3, 4, 5, 6 |
| 2 days before | Optional: start the live-inference server (`python -m omari_us.serve --config configs/export.yaml`) and confirm the closing chapter reports "connected" | Live inference available |
| Day of | Share the participant URL (static host, or the laptop's LAN address via `npm run preview -- --host`) | Participants can load it on laptops and tablets |
| Day of | Projector: 1920×1080 or larger, browser zoom 110–125% | Body text readable from the back row |

**Fallback.** If no bundle is available, the workshop still runs. All live simulations work, and model-output figures show an explicit "Awaiting artifact" state. Teach those segments as "here is what we will measure, and why", using the toy models in Chapters 5–6, and spend more time on the data and architecture labs.

### Presenter controls

`P` toggles presenter mode (larger type, HUD with timer and cue). `←`/`→` step between figures and headings. `[`/`]` jump chapters. `F` toggles fullscreen and `T` starts or pauses the clock. The top bar is the session timeline: segment width = minutes, and the dark marker is elapsed time once the clock runs. Open `?mode=present` to start in presenter mode.

## Running order

### 0:00–0:10 · Chapter 1: Can AI recognise this ultrasound plane?

| Min | Do | Say / ask |
|---|---|---|
| 0:00 | Title block on screen. Start the clock (`T`) | Introduce the central question: how do different ways of learning image representations change prediction, generalisation and explainability? |
| 0:02 | Opening figure. Ask for a show of hands for each plane **before** anyone clicks | "What would you call this plane, and what features did you use?" |
| 0:04 | Pick the majority answer, click **Reveal** | Read the expert label, then the three probability bars. Point to differences in confidence even when all three are correct |
| 0:06 | Step through two disagreement images (arrows) | "Same image, three models, three answers. What could explain that?" Collect 2–3 hypotheses (anatomy, image quality, overlays, zoom) and write them down; you will return to them in Chapter 6 |
| 0:08 | Read the paragraph on correctness, confidence and reasoning | Accuracy measures the first, calibration the second; the third has no single metric |
| 0:09 | Pull-question | "If all three predict correctly, do we know they learned the right features?" Leave it open |

### 0:10–0:25 · Chapter 2: From ultrasound pixels to training data

| Min | Do | Say / ask |
|---|---|---|
| 0:10 | Text on speckle, dynamic range, resolution, acquisition | Speckle is multiplicative and deterministic, not sensor noise; images also carry operator and machine choices |
| 0:12 | **Preprocessing lab**. Start with the phantom, then switch to an exported sample | Increase speckle with L = 1, then raise L to 8: variance falls as 1/L. Reduce resolution to ¼ and point to the patch grid in the 224 input |
| 0:15 | Geometry: Stretch → Letterbox → Centre crop | Read the scale numbers aloud. Stretching is what the official USF-MAE loader does. Ask: "Could aspect ratio become a class cue?" |
| 0:17 | Normalisation: ImageNet → per-image z → CLAHE | The histogram on the right is the tensor the network sees. If the bundle has perturbations, click an "Exported settings" chip and show the predictions change |
| 0:19 | Ask the multiple-choice question on stretching; reveal | — |
| 0:20 | **Class balance** figure. Switch weighting None → Inverse → Effective number | "Other" contributes most of the unweighted loss. Preview macro-F1 |
| 0:22 | **Leakage** figure. Slide the test fraction; move "near-duplicate frames" from 1 to 8 | The left number is exact arithmetic. The right panel is a toy, but the gap is the leakage. The authors' split is patient-level and temporal; ours keeps it |
| 0:24 | Transition | "We have a clean tensor and a clean split. Now: what do the three models do with it?" |

### 0:25–0:50 · Chapter 3: Inside the models (9 + 9 + 7 min)

| Min | Do | Say / ask |
|---|---|---|
| 0:25 | Model comparison table | Same task, three recipes: architecture **and** pretraining data **and** objective differ. This is not a controlled ablation |
| 0:27 | **3A Kernel explorer**: Sobel x → Laplacian → edit one weight | Weight sharing makes it translation-equivariant. With ReLU on, only positive evidence survives |
| 0:30 | Change stride and dilation | Read the output-size formula in the caption |
| 0:31 | **Receptive field**: click blocks 1 → 4 | Exact theoretical RF; by block 3 it exceeds the image. The effective RF is much smaller |
| 0:33 | **Dense block**: hover H4, change k | Concatenation, not addition. 2208 channels at the end; GAP + linear head (remember for CAM) |
| 0:34 | **Real feature maps** (if exported): denseblock1 → 4 | Early: edges and speckle. Late: coarse, class-selective. Single channels are rarely interpretable |
| 0:36 | **3B Patch explorer**: click several patches | 196 tokens + [CLS], each patch = 768 numbers. The right panel is MAE's fixed sin-cos table. The supervised ViT learns its own |
| 0:39 | Open *Mathematical details* briefly for the technical half of the room | Attention equation, d_h = 64 |
| 0:40 | **Attention explorer**: start in Toy mode; set position weight 0 → 3 | Locality has to be learned or injected. Then switch to Model mode: block 1 → 12, mean → individual heads, [CLS] → patch query |
| 0:43 | Read the paragraph "Attention maps are seductive…" | Plant the distinction you will need in Chapter 5 |
| 0:45 | **3C Masking explorer**: ratio 25% → 75% → 90% | The encoder sees 49 of 196 patches at 75%. Compute drops to ~25%. Show the authentic USF-MAE reconstruction if exported; contrast with the naive fill |
| 0:47 | Pretraining vs fine-tuning schematic | Decoder discarded; new 6-way head. Frozen = linear probe |
| 0:48 | **Overlap callout**: read aloud | FETAL_PLANES_DB was part of OpenUS-46 (images, not labels). State it plainly |
| 0:49 | Ask the multiple-choice question; reveal | Only "it depends" is defensible before measuring |

### 0:50–1:10 · Chapter 4: Training, evaluation and generalisation

| Min | Do | Say / ask |
|---|---|---|
| 0:50 | Text on frozen vs fine-tuning; open *Mathematical details* (weighted CE, focal, layer decay) | Same splits, loss and selection rule; each model keeps its own established recipe |
| 0:53 | **Test-set performance**: accuracy vs macro-F1 vs balanced accuracy; CIs | Patient-cluster bootstrap. Note how wide the intervals are |
| 0:56 | Click the abdomen→thorax cell | Look at the images together: are these model errors or label ambiguity? |
| 0:58 | Per-class metric toggle F1 → precision → recall | Which class is hardest for each model? |
| 0:59 | **ROC / threshold**: positive class = Brain, slide the threshold | A threshold only makes sense for a one-vs-rest question |
| 1:01 | **Prevalence simulator**: dataset mix → cervical-length clinic | Recall unchanged; precision moves. This is why deployment setting matters |
| 1:04 | **Calibration**: reliability diagram; temperature what-if | T is fitted on validation in notebook 05; here it is for intuition. Accuracy is unchanged by T |
| 1:06 | **Label efficiency** curves | Where (if anywhere) does domain pretraining help most? Check seeds and bands before reading differences |
| 1:08 | **By machine** table | Subgroups are confounded; leave-one-machine-out is the stronger test |
| 1:09 | Final multiple-choice question; reveal | Repeat both caveats: recipes differ; pretraining overlap |

### 1:10–1:20 · Break

Leave the page on the Break section. Its three prompts point participants to labs they can explore alone. Restart the clock at 1:20.

### 1:20–1:45 · Chapter 5: Explainability, from CAM to MetaCAM++

| Min | Do | Say / ask |
|---|---|---|
| 1:20 | Why explanations: debug, discover, support trust | They answer a narrow spatial question, not "why" |
| 1:22 | CAM equation | Exact for DenseNet's GAP + linear head |
| 1:24 | Grad-CAM, Grad-CAM++ equations | Grad-CAM generalises CAM; Grad-CAM++ re-weights positive gradients |
| 1:26 | **Toy CAM**: "Anatomy-like weights"; switch CAM → Grad-CAM → Grad-CAM++ | You know the truth here because you set the weights. Point to the deletion curve vs random |
| 1:29 | Drag the "overlay" weight up | The map moves to the burned-in text: the explanation can reveal a shortcut **when the model uses one** |
| 1:30 | Open *Limitations*: resolution (7×7), ReLU, normalisation hiding magnitude | — |
| 1:32 | **Explainability lab** (real maps): one correct and one incorrect sample; compare methods for DenseNet | Note how much methods differ on the same model and image |
| 1:36 | Switch to "Models for one method" | Coarse 7×7 vs 14×14 maps; different models may "look" at different regions with the same prediction |
| 1:38 | Change target class to a wrong class | Class-specific maps change; attention rollout does not |
| 1:40 | Colour scale → jet | Why perceptually uniform maps matter |
| 1:41 | Transformer text: why CAM doesn't transfer; token reshape; rollout | Attention routing ≠ class attribution |
| 1:43 | Compatibility table, then MetaCAM++ | MetaCAM++ is research in progress. Explain the plug-in interface; promise details when available. Do not speculate about results |

### 1:45–1:55 · Chapter 6: Model detective

| Min | Do | Say / ask |
|---|---|---|
| 1:45 | Shortcut-learning paragraph | List candidate shortcuts in obstetric ultrasound |
| 1:46 | **Detective lab**: pick a sample; the room votes on hypotheses; tick them; **Lock** | Commit before seeing evidence |
| 1:48 | Reveal: read the perturbation table row by row | Which predictions flip? Which maps stay stable (ρ)? Compare with the hypotheses from 0:06 |
| 1:51 | Annotation detector slider | What fraction of pixels are overlays? Was inpainting them enough to change anything? |
| 1:52 | Plausible vs faithful paragraph; **shortcut toy** | A map can look right and be wrong, and vice versa |
| 1:53 | **Sanity check**: draw new random weights several times | High ρ means the map mostly reflects the image, not the model |
| 1:54 | Occlusion question; reveal | Influence ≠ correctness; occluders are out of distribution |

### 1:55–2:00 · Chapter 7: What would you trust?

| Min | Do | Say / ask |
|---|---|---|
| 1:55 | Opening image again | "Has your view of these three predictions changed?" |
| 1:56 | Model question; deployment checklist ticked live with the room | Most items cannot be shown on one public dataset, and that is the point |
| 1:58 | Five takeaways; notebooks list | Point to the notebooks; demo mode runs on a free Colab GPU |
| 1:59 | Thank, questions | — |

## Timing buffers

- **Running late by 5 min at 0:50:** skip the ROC figure and the last Chapter 4 question.
- **Running late at 1:20:** shorten the toy CAM to one method switch, and skip the jet colour-map demo.
- **Ahead of time:** run the sanity check live on real checkpoints via notebook 06, or open *View PyTorch implementation* panels for the technical half of the room.

## Questions to expect, and honest answers

- **"Which model is best?"** Show the CIs and the label-efficiency curve. If differences overlap, say so. Mention the pretraining overlap for USF-MAE.
- **"Is the heatmap what the model sees?"** No. It shows where a particular perturbation-free gradient signal is large at 7×7 or 14×14 resolution. Use the deletion curve and sanity check as evidence.
- **"Could this be deployed?"** Not on this evidence: single public dataset, two hospitals, no prospective or external validation (see the checklist).
- **"Why not train on everything at full resolution?"** Pretrained weights expect 224². Higher resolution is possible (interpolated positional embeddings for ViTs) but changes compute and comparability.
- **"Is MetaCAM++ better?"** Not presented yet; the slot exists so it can be evaluated with the same faithfulness tests.
