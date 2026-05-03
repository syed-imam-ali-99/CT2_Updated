#!/bin/bash
# ==============================================================================
# CT2 Colorization Transformer - Inference (Colorize Images)
# ==============================================================================
# Usage: bash scripts/inference.sh
#
# Prerequisites:
#   1. Set dataset_dir in configs/paths.yaml (ImageNet val images)
#      OR pass --dataset_dir to override
#   2. Place trained checkpoint as <LOG_DIR>/checkpoint.pth
#   3. Colorized outputs saved to OUTPUT_DIR
#
# To use on custom images instead of ImageNet val:
#   Set DATASET="random" and DATASET_DIR to your image folder
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Configurable parameters ----
LOG_DIR="segm/vit-large"           # Directory containing checkpoint.pth
OUTPUT_DIR="results/imagenet"      # Where colorized images will be saved
BACKBONE="vit_large_patch16_384"   # Model backbone
DATASET="random"                   # "coco" for ImageNet val, "random" for custom folder
DATASET_DIR=""                     # Leave empty to use configs/paths.yaml
GPU=0

# ---- Run inference ----
uv run python -m torch.distributed.launch \
    --nproc_per_node=1 \
    --master_port=12345 \
    test.py \
    --log-dir "${LOG_DIR}" \
    --output-dir "${OUTPUT_DIR}" \
    --backbone "${BACKBONE}" \
    --dataset "${DATASET}" \
    --dataset_dir "${DATASET_DIR}" \
    --local_rank ${GPU} \
    --only_test True
