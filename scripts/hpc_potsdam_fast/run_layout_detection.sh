#!/usr/bin/env bash
set -euo pipefail

NEWSPAPER="${1:?Usage: $0 <newspaper> <detector> [workers] [sub_dirs...]}"
DETECTOR="${2:?Usage: $0 <newspaper> <detector> [workers] [sub_dirs...]}"
WORKERS="${3:-1}"
shift 3 || true

SUB_DIR_ARGS=()
if [ "$#" -gt 0 ]; then
  SUB_DIR_ARGS=(--sub-dirs "$@")
fi

export PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True
export PADDLE_PDX_MODEL_SOURCE=HuggingFace
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export CPU_NUM=1

.venv/corpus_construction/layout_detection/bin/python -m src.workflows.layout_detection \
  --preprocessed-dir "$HOME/data/leonrios/processed/enhanced/$NEWSPAPER" \
  --output-dir "$HOME/data/leonrios/processed/layout/$NEWSPAPER" \
  --newspaper "$NEWSPAPER" \
  --detectors "$DETECTOR" \
  --workers "$WORKERS" \
  "${SUB_DIR_ARGS[@]}"