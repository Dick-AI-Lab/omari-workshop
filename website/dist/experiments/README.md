# Artifact bundle folder

The website loads `manifest.json` from this folder. When it is absent, every model-output figure shows an "Awaiting artifact" state and the live simulations still work.

## How this folder is populated

Everything here except this README is **generated** and **git-ignored** (see `.gitignore`). Build it from the Colab outputs in `content/omari_artifacts/omari_artifacts/`:

    python scripts/finalize_website_bundle.py --artifacts content/omari_artifacts/omari_artifacts --out website/public/experiments
    # optional: official 500-epoch MAE reconstructions (GPU recommended)
    python scripts/finalize_website_bundle.py --artifacts content/omari_artifacts/omari_artifacts --out website/public/experiments --reconstruct --device cuda
    python scripts/check_provenance.py website/public/experiments

The finalize script writes schema 1.1 (`docs/ARTIFACT_SCHEMA.md`). It reads the original Colab bundle (`ART/website_bundle/`), the runs, the splits, `faithfulness.csv` and `dataset_stats.json`, and leaves them untouched. Do **not** copy `ART/website_bundle/` or `ART/website_export/` here by hand: they have known defects (`docs/PROVENANCE.md`). The provenance of every file is described in `docs/PROVENANCE.md`.

Never copy smoke-test bundles (random weights, synthetic images) here.

## Deployment notes

- The bundle is about 70 MB and contains CC BY 4.0 FETAL_PLANES_DB images, so it is not committed. Attribution is in the manifest and the site footer.
- Vite copies `public/` into `dist/` at build time. Build the bundle first, then run `cd website && npm ci && npm run build`. The resulting `website/dist/` (including `dist/experiments/`) can be served from any static host, or previewed with `npx vite preview`.
- For a host that deploys from git (e.g. GitHub Pages via CI), add a step that runs the finalize script or downloads a released bundle archive into this folder before `npm run build`. Do not force-add the bundle to the repository.
- After replacing the bundle, hard-reload the browser. The site fetches `experiments/manifest.json` at load time and accepts schema `1.0` and `1.1`.
