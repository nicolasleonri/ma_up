#!/usr/bin/env bash
set -euo pipefail

NEWSPAPER="${1:?Usage: $0 <newspaper> [workers] [sub_dirs...]}"
WORKERS="${2:-6}"
shift 2 || shift 1

SUB_DIR_ARGS=()
if [ "$#" -gt 0 ]; then
  SUB_DIR_ARGS=(--sub-dirs "$@")
fi

BASE_DIR="$HOME/data/leonrios/processed"

case "$NEWSPAPER" in
  elcomercio|gestion)
    INPUT_DIR="$BASE_DIR/enhanced/$NEWSPAPER"
    ;;
  *)
    INPUT_DIR="$BASE_DIR/layout/$NEWSPAPER"
    ;;
esac

OUTPUT_DIR="$BASE_DIR/binarized/$NEWSPAPER"

.venv/corpus_construction/binarize/bin/python -m src.workflows.binarize \
  --input-dir "$INPUT_DIR" \
  --output-dir "$OUTPUT_DIR" \
  --newspaper "$NEWSPAPER" \
  --workers "$WORKERS" \
  "${SUB_DIR_ARGS[@]}"