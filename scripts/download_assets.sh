#!/usr/bin/env bash
# Download the external assets this repository does not ship. Run from the repository root.
#   bash scripts/download_assets.sh [DATA_DIR]
# Environment options:
#   INSTALL_GDOWN=1          pip-install gdown if it is missing (default: do not touch the environment; gdown>=5.1 is in requirements.txt)
#   WITH_HISTORICAL_100EP=1  also download the historical 100-epoch USF-MAE checkpoint (only needed to reproduce the
#                            reconstructions of the original website bundle; not used by any reported result)
set -euo pipefail
DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR" checkpoints

echo "1/2 FETAL_PLANES_DB (2.1 GB, CC BY 4.0) from Zenodo record 3904280"
if [ ! -f "$DATA_DIR/FETAL_PLANES_ZENODO.zip" ]; then
  curl -L --fail -o "$DATA_DIR/FETAL_PLANES_ZENODO.zip" "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1"
fi
echo "2a5fcc2cefb789bcc0f6c1f73e0ea43f  $DATA_DIR/FETAL_PLANES_ZENODO.zip" | md5sum -c -
unzip -n -q "$DATA_DIR/FETAL_PLANES_ZENODO.zip" -d "$DATA_DIR"

echo "2/2 USF-MAE official 500-epoch checkpoint usf_mae_500ep.pth (Google Drive, via gdown)"
if ! command -v gdown >/dev/null 2>&1; then
  if [ "${INSTALL_GDOWN:-0}" = "1" ]; then
    python -m pip install -q "gdown>=5.1"
  else
    echo "gdown not found. Install it (pip install 'gdown>=5.1', listed in requirements.txt) or re-run with INSTALL_GDOWN=1." >&2
    exit 1
  fi
fi
# gdown >= 5 removed --id; pass the file id positionally.
[ -f checkpoints/usf_mae_500ep.pth ] || gdown 15n71qHXSBjEjrLtCOzvBb6gTsGhh7NUg -O checkpoints/usf_mae_500ep.pth
CKPTS="checkpoints/usf_mae_500ep.pth"
if [ "${WITH_HISTORICAL_100EP:-0}" = "1" ]; then
  echo "    (historical) USF-MAE 100-epoch checkpoint usf_mae_100ep.pth"
  [ -f checkpoints/usf_mae_100ep.pth ] || gdown 1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr -O checkpoints/usf_mae_100ep.pth
  CKPTS="$CKPTS checkpoints/usf_mae_100ep.pth"
fi

# Verify structure and the pinned sha256 (a mismatch is fatal here).
OMARI_STRICT_CHECKPOINT_SHA=1 python - $CKPTS <<'PY'
import sys; sys.path.insert(0, "python")
from omari_us.models.usf_mae import load_usf_mae_checkpoint
for path in sys.argv[1:]:
    _, rep = load_usf_mae_checkpoint(path)
    print(rep.summary())
PY
echo "DenseNet-161 and ViT-B/16 ImageNet weights download automatically on first use (torchvision / timm via Hugging Face)."
