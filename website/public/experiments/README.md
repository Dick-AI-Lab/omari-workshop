# Artifact bundle folder

Empty on purpose. The website loads `manifest.json` from here when it exists; otherwise every model-output figure shows an "Awaiting artifact" state.

Fill it with real outputs:

    python scripts/export_website_artifacts.py --config configs/export.yaml --clean
    python scripts/check_provenance.py website/public/experiments

Never copy smoke-test bundles (random weights, synthetic images) here. Schema: docs/ARTIFACT_SCHEMA.md.
