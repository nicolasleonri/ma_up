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

.venv/corpus_construction/layout_detection/bin/python -m src.workflows.layout_detection \
  --preprocessed-dir "data/processed/enhanced/$NEWSPAPER" \
  --output-dir "data/processed/layout/$NEWSPAPER" \
  --newspaper "$NEWSPAPER" \
  --detectors "$DETECTOR" \
  --workers "$WORKERS" \
  "${SUB_DIR_ARGS[@]}"