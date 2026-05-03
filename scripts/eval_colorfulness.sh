#!/bin/bash
# ==============================================================================
# CT2 Colorization Transformer - Evaluate Colorfulness
# ==============================================================================
# Usage: bash scripts/eval_colorfulness.sh [--pred_dir DIR]
# ==============================================================================

set -e

PROJECT_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$PROJECT_ROOT"

# ---- Default directory (override via CLI arg or edit here) ----
PRED_DIR="${PRED_DIR:-output/colorized}"

# Override with CLI args if provided
while [[ $# -gt 0 ]]; do
    case $1 in
        --pred_dir) PRED_DIR="$2"; shift 2 ;;
        *)          shift ;;
    esac
done

echo "Evaluating colorfulness..."
echo "  Predictions: ${PRED_DIR}"

python -m segm.eval_colorfullness \
    --pred_dir "${PRED_DIR}"
