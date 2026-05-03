#!/bin/bash
# ==============================================================================
# CT2 Colorization Transformer - Evaluate SSIM, PSNR, LPIPS
# ==============================================================================
# Usage: bash scripts/eval_metrics.sh [--pred_dir DIR] [--gt_dir DIR]
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Default directories (override via CLI args or edit here) ----
PRED_DIR="${PRED_DIR:-output/colorized}"
GT_DIR="${GT_DIR:-/path/to/ImageNet/val}"

# Override with CLI args if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --pred_dir) PRED_DIR="$2"; shift 2 ;;
        --gt_dir)   GT_DIR="$2";   shift 2 ;;
        *)          shift ;;
    esac
done

echo "Evaluating metrics..."
echo "  Predictions: ${PRED_DIR}"
echo "  Ground truth: ${GT_DIR}"

python -m segm.eval_metrics \
    --pred_dir "${PRED_DIR}" \
    --gt_dir "${GT_DIR}"
