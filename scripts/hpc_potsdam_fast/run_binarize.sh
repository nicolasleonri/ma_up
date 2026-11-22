#!/usr/bin/env bash
set -euo pipefail

NEWSPAPER="${1:?Usage: $0 <newspaper> [workers] [sub_dirs...]}"
WORKERS="${2:-6}"
shift 2 || shift 1

SUB_DIR_ARGS=()
if [ "$#" -gt 0 ]; then
  SUB_DIR_ARGS=(--sub-dirs "$@")
fi

uv run python -m src.workflows.binarize \
  --input-dir "$HOME/data/leonrios/processed/layout/$NEWSPAPER" \
  --output-dir "$HOME/data/leonrios/processed/enhanced/$NEWSPAPER" \
  --newspaper "$NEWSPAPER" \
  --workers "$WORKERS" \
  "${SUB_DIR_ARGS[@]}"