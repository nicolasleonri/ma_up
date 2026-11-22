#!/usr/bin/env bash
set -euo pipefail

NEWSPAPER="${1:?Usage: $0 <newspaper> [workers] [sub_dirs...]}"
WORKERS="${2:-4}"
shift 2 || true

SUB_DIR_ARGS=()
if [ "$#" -gt 0 ]; then
  SUB_DIR_ARGS=(--sub-dirs "$@")
fi

.venv/corpus_construction/enhance_images/bin/python -m src.workflows.enhance_images \
  --input-dir "$HOME/data/leonrios/$NEWSPAPER" \
  --output-dir "$HOME/data/leonrios/processed/enhanced/$NEWSPAPER" \
  --newspaper "$NEWSPAPER" \
  --workers "$WORKERS" \
  "${SUB_DIR_ARGS[@]}"