#!/usr/bin/env bash
set -euo pipefail

NEWSPAPER="${1:?Usage: $0 <newspaper> [sub_dirs...]}"
shift 1

SUB_DIR_ARGS=()
if [ "$#" -gt 0 ]; then
  SUB_DIR_ARGS=(--sub-dirs "$@")
fi

export HF_HOME="$HOME//data/huggingface"
BASE_DIR="$HOME/data/leonrios/processed/"
PROMPTS_DIR="$HOME/project/leonrios/ma_up/data/corpus_construction/local_dspy_llm/prompts"

case "$NEWSPAPER" in
  elcomercio|gestion)
    MODE="none"
    ;;
  *)
    MODE="cropped"
    ;;
esac

uv run python -m src.workflows.llm_extraction \
  --newspaper "$NEWSPAPER" \
  --prompts-dir "$PROMPTS_DIR" \
  --ocr-dir "$BASE_DIR/ocr_extraction/$NEWSPAPER/$MODE" \
  --output-dir "$BASE_DIR/llm_extraction/$NEWSPAPER/$MODE" \
  --gpu-memory-utilization 0.85 \
  "${SUB_DIR_ARGS[@]}"