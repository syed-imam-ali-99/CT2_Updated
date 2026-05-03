#!/bin/bash
# ==============================================================================
# CT2 Colorization Transformer - Generate Color Quantization Ground Truth
# ==============================================================================
# Usage: bash scripts/make_q_actual.sh [--dataset_dir DIR] [--split SPLIT]
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Default parameters (override via CLI args or edit here) ----
DATASET_DIR="${DATASET_DIR:-/path/to/ImageNet}"
SPLIT="${SPLIT:-val}"

# Override with CLI args if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --dataset_dir) DATASET_DIR="$2"; shift 2 ;;
        --split)       SPLIT="$2";       shift 2 ;;
        *)             shift ;;
    esac
done

echo "Generating q_actual..."
echo "  Dataset: ${DATASET_DIR}"
echo "  Split:   ${SPLIT}"

python -m segm.make_q_actual \
    --dataset_dir "${DATASET_DIR}" \
    --split "${SPLIT}"
