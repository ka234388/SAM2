#!/usr/bin/env bash
set -euo pipefail

# --- CONFIG ---
ENV_NAME="cvs_sam2"
PYTHON_VERSION="3.10"
REQ_FILE="requirements.txt"
OUT_ROOT="/kaggle/working/a3_sam2_camvid"   # where CSV + masks are saved
DATA_ROOT="/kaggle/input/camvid/CamVid"     # dataset root (with val/ and val_labels/)

# --- Conda setup ---
# Load conda (path may differ: ~/miniconda3 or ~/anaconda3)
source ~/anaconda3/etc/profile.d/conda.sh || source ~/miniconda3/etc/profile.d/conda.sh

# Create env if it doesn't exist
if ! conda env list | grep -q "$ENV_NAME"; then
    echo "[INFO] Creating conda env: $ENV_NAME"
    conda create -n "$ENV_NAME" python=$PYTHON_VERSION -y
fi

# Activate env
echo "[INFO] Activating env: $ENV_NAME"
conda activate "$ENV_NAME"

# Install requirements
echo "[INFO] Installing Python dependencies"
pip install -r "$REQ_FILE"

# --- Run evaluate.py ---
echo "[INFO] Running evaluate.py ..."
python evaluate.py \
  --out-root "$OUT_ROOT" \
  --data-root "$DATA_ROOT" \
  --samples 5 \
  --mode top_people_gain \
  --overlays

echo "Done. Output is saved under location $OUT_ROOT"
