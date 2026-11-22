#!/usr/bin/env bash
set -euo pipefail

NEWSPAPER="${1:?Usage: $0 <newspaper> [workers] [batch_size] [sub_dirs...]}"
WORKERS="${2:-1}"
BATCH_SIZE="${3:-8}"
shift 3 || shift 2 || shift 1

SUB_DIR_ARGS=()
if [ "$#" -gt 0 ]; then
  SUB_DIR_ARGS=(--sub-dirs "$@")
fi

BASE_DIR="$HOME/data/leonrios"
BINARIZED_DIR="$BASE_DIR/processed/binarized/$NEWSPAPER"

case "$NEWSPAPER" in
  elcomercio|gestion)
    OUTPUT_DIR="$BASE_DIR/processed/ocr/$NEWSPAPER/none"
    ;;
  *)
    OUTPUT_DIR="$BASE_DIR/processed/ocr/$NEWSPAPER/cropped"
    ;;
esac

.venv/corpus_construction/ocr_extraction/bin/python -m src.workflows.ocr_extraction \
  --newspaper "$NEWSPAPER" \
  --binarized-dir "$BINARIZED_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --workers "$WORKERS" \
  --batch-size "$BATCH_SIZE" \
  "${SUB_DIR_ARGS[@]}"