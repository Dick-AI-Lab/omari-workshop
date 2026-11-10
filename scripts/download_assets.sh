#!/usr/bin/env bash
# Download the external assets this repository does not ship. Run from the repository root.
#   bash scripts/download_assets.sh [DATA_DIR]
set -euo pipefail
DATA_DIR="${1:-data}"
mkdir -p "$DATA_DIR" checkpoints
echo "1/2 FETAL_PLANES_DB (2.1 GB, CC BY 4.0) from Zenodo record 3904280"
if [ ! -f "$DATA_DIR/FETAL_PLANES_ZENODO.zip" ]; then
  curl -L -o "$DATA_DIR/FETAL_PLANES_ZENODO.zip" "https://zenodo.org/records/3904280/files/FETAL_PLANES_ZENODO.zip?download=1"
fi
echo "2a5fcc2cefb789bcc0f6c1f73e0ea43f  $DATA_DIR/FETAL_PLANES_ZENODO.zip" | md5sum -c -
unzip -n -q "$DATA_DIR/FETAL_PLANES_ZENODO.zip" -d "$DATA_DIR"
echo "2/2 USF-MAE 100-epoch checkpoint (Google Drive, via gdown)"
pip install -q gdown
[ -f checkpoints/usf_mae_100ep.pth ] || gdown --id 1ZPu_7KhMEuaq-XdLhVp2EEgMgLJ4dKhr -O checkpoints/usf_mae_100ep.pth
python - <<'PY'
import sys; sys.path.insert(0, "python")
from omari_us.models.usf_mae import load_usf_mae_checkpoint
_, rep = load_usf_mae_checkpoint("checkpoints/usf_mae_100ep.pth")
print(rep.summary())
PY
echo "DenseNet-161 and ViT-B/16 ImageNet weights download automatically on first use (torchvision / timm via Hugging Face)."
