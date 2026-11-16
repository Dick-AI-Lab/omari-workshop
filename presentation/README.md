# OMARI seminar deck: *From Pixels to Predictions*

The editable PowerPoint for the OMARI seminar *From Pixels to Predictions: Developing AI Models for Medical Image Interpretation* (Kevin Dick, Youssef Megahed). It is designed to be presented together with the interactive workshop website in `website/`. The deck tells the story; the website is where participants explore the results hands-on.

## Contents

| Path | What it is |
|---|---|
| `OMARI_From_Pixels_to_Predictions.pptx` | **Main deck.** 16:9, 37 slides: 35 timed slides (120 min including the 10-min break) plus 2 appendix slides. Every slide has speaker notes. Text, diagrams, tables and two charts are native, editable PowerPoint objects |
| `PRESENTER_GUIDE.md` | Run of show (slide · clock time · minutes · website cue), discussion prompts, and the full speaker notes for every slide. Generated from the same specification as the deck |
| `assets/figures/` | PNG figures made from the saved model outputs and dataset images (see `assets/figure_provenance.json`) |
| `assets/equations/` | Equations rendered from LaTeX source (the source strings are in `src/make_figures.py`) |
| `assets/figure_provenance.json` | For each figure: source sample, checkpoint hash, method and layer, and the numbers quoted |
| `preview/OMARI_From_Pixels_to_Predictions.pdf` | PDF export by Microsoft PowerPoint |
| `preview/slides/slide_NN.png`, `preview/contact_sheet.png` | 1920×1080 renders of every slide, and a one-page overview |
| `preview/validation.json` | Result of the automated checks (`src/validate_deck.py`) |
| `src/` | Everything needed to regenerate the deck (below) |

## How the PowerPoint and the website split the work

- **Slides:** the narrative, the concepts, the key equations, the real headline results and the caveats. The deck works on its own if the website is unavailable.
- **Website:** hands-on exploration. Six teal "Explore on the workshop website" slides (Chapters 1–6; Chapter 7 and the break slide point to the site as well) each give the chapter, three steps, a question to investigate, a duration and a discussion prompt.
- **No website URL was found in the repository.** The activity slides therefore show an editable placeholder (`[Workshop website URL: add before the session]`) and a dashed QR box. To fill both, rebuild with `--url` (below). Localhost addresses are refused.

## What is real and what is illustrative

Every figure carries a badge, matching the website's provenance badges:

| Badge | Meaning |
|---|---|
| **MODEL OUTPUT** (green) | Saved outputs of this project's trained models: feature maps, attention, attributions, 500-epoch USF-MAE reconstructions, perturbation predictions, and all metrics. Read from `website/public/experiments/` (bundle schema 1.1); nothing is re-run |
| **DATASET IMAGE / METADATA** (blue) | FETAL_PLANES_DB images (CC BY 4.0), and counts computed from the dataset CSV and the split manifest |
| **ILLUSTRATIVE DIAGRAM** (ochre) | Schematic diagrams drawn for the deck (convolution, dense block, patch embedding, MAE pipeline, workflow, Grad-CAM pipeline, patient split) |

All USF-MAE reconstructions come from `usf_mae_500ep.pth` (sha256 `c659a061…b3a2533`). `make_figures.py` asserts this, and the 100-epoch checkpoint is not used anywhere in the deck.

## Regenerating the deck

The build needs Python 3.12 and the finalized website bundle in `website/public/experiments/` (`scripts/finalize_website_bundle.py`). It uses a separate virtual environment so that the project environment is not touched:

```powershell
cd presentation
python -m venv --system-site-packages .venv          # reuses numpy / matplotlib / Pillow / lxml
.venv\Scripts\python -m pip install python-pptx==1.0.2 qrcode==7.4.2
.venv\Scripts\python src\make_figures.py             # figures + equations from the bundle
.venv\Scripts\python src\build_deck.py               # deck + PRESENTER_GUIDE.md
.venv\Scripts\python src\build_deck.py --url https://<public-site>   # same, with URL and QR codes
.venv\Scripts\python src\validate_deck.py            # 58 automated checks
powershell -ExecutionPolicy Bypass -File src\render_preview.ps1      # PDF, PNGs, contact sheet (needs PowerPoint)
```

`build_deck.py` overwrites `OMARI_From_Pixels_to_Predictions.pptx` and `PRESENTER_GUIDE.md`. **If you have edited the deck by hand in PowerPoint, save it under another name first.** Edits are better made in `src/build_deck.py`, where each slide is one block of code next to its speaker notes, so that the deck and the guide stay in sync.

### Editing in PowerPoint

All text, diagrams, tables, the dataset chart and the calibration chart are editable. The label-efficiency chart and the heatmap panels are images, because they show individual seeds and pixel data. Equations are high-resolution images; their LaTeX is in `src/make_figures.py`. Fonts are Georgia (titles), Segoe UI (text) and Consolas, all standard on Windows. On macOS, Segoe UI is substituted; check line breaks after opening it there.

## Validation performed

- **Render.** The deck was opened and exported by Microsoft PowerPoint (COM automation) to PDF and to 1920×1080 PNGs. Every slide was inspected visually, and the clipping and overlap issues found (an equation, the faithfulness table, the break-slide URL, the dense-block wiring, and a confusion-matrix header) were fixed and re-rendered.
- **Automated (`validate_deck.py`, re-reading the saved .pptx).**
  - Exact 16:9 size; no shape outside the slide; speaker notes on every slide; embedded images at least 90 effective ppi.
  - Every headline number on the slides and in the notes matches the bundle JSON: full-label accuracy and macro-F1, the USF-MAE 3-seed mean ± sd, ECE, T, McNemar p, the 5% label-efficiency values, dataset counts, the confusion error count, all faithfulness rows, and the opening-image probabilities.
  - 500-epoch reconstruction provenance, no localhost URL, terminology guards, and a total of exactly 120 minutes.
- **QR path.** Tested with a dummy URL; the generated code decodes correctly with OpenCV.
- **Not verified:** projector legibility in a real room, rendering in Keynote, Google Slides or macOS PowerPoint, and the final public URL.

## Reuse, licensing and sharing restrictions

- **FETAL_PLANES_DB images** are CC BY 4.0: attribution is required and is on the slides and in the appendix (Burgos-Artizzu et al., *Sci. Rep.* 2020; Zenodo 3904280).
- **USF-MAE checkpoints have no stated licence**, and they derive from Facebook MAE weights released under CC BY-NC 4.0. The USF-MAE reconstructions (slide 13) and the USF-MAE attention and attribution maps are outputs of those weights.
  - Treat the deck as material for **non-commercial research and teaching** until the checkpoint licence is clarified.
  - Check with the USF-MAE authors before any public or commercial redistribution, such as posting the deck or PDF online.
- The ImageNet-initialised DenseNet-161 and ViT-B/16 weights carry their own terms (see `docs/DATA_AND_MODEL_SOURCES.md`).
- **Nothing in this folder has been published or distributed.**

## Scientific caveats carried by the deck

These caveats appear on the slides or in the notes:
- The pretraining-overlap disclosure for USF-MAE (it is unlabelled exposure to test pixels, not label leakage).
- "No significant difference" is not equivalence.
- No significance test was run for the low-label results, and ViT-B/16 below 100% was not run.
- Accuracy and macro-F1 are reported separately.
- Attention is not attribution, and plausible is not faithful.
- The faithfulness scores are based on n = 30 images.
- The DenseNet-161 checkpoint history is covered in appendix slide 37 and `docs/PROVENANCE.md`.
