#!/bin/bash
# ==============================================================================
# CT2 Colorization Transformer - Training Script (Multi-GPU)
# ==============================================================================
# Usage: bash scripts/train.sh
# All paths are configured in configs/paths.yaml
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Configurable parameters ----
NUM_GPUS=8                         # Number of GPUs
BACKBONE="vit_large_patch16_384"
DECODER="mask_transformer"
DATASET="coco"
BATCH_SIZE=48                      # Total batch size across all GPUs
LOG_DIR="segm/vit-large"
EPOCHS=256
LR=0.001

# Loss options
ADD_MASK=True
ADD_L1=True
L1_WEIGHT=10
L1_CONV=True
COLOR_POSITION=True
ADD_CONV=True

# Uncomment to enable
# AMP="--amp"
# PARTIAL_FT="--partial_finetune True"
# ADD_EDGE="--add_edge True --edge_loss_weight 0.05"

# ---- Run training ----
python -m torch.distributed.launch \
    --nproc_per_node=${NUM_GPUS} \
    train.py \
    --log-dir "${LOG_DIR}" \
    --backbone ${BACKBONE} \
    --decoder ${DECODER} \
    --dataset ${DATASET} \
    --batch-size ${BATCH_SIZE} \
    --epochs ${EPOCHS} \
    -lr ${LR} \
    --local_rank 0 \
    --add_mask ${ADD_MASK} \
    --add_l1_loss ${ADD_L1} \
    --l1_weight ${L1_WEIGHT} \
    --l1_conv ${L1_CONV} \
    --color_position ${COLOR_POSITION} \
    --add_conv ${ADD_CONV} \
    ${AMP:-} \
    ${PARTIAL_FT:-} \
    ${ADD_EDGE:-}
