#!/bin/bash
set -euo pipefail
NEWSPAPER=$1; VARIANT=$2; LLM=$3
cd ~/project/leonrios/ma_up
source .venv/corpus_construction/llm_extraction/bin/activate
export VLLM_USE_FLASHINFER_SAMPLER=0
# export HF_HOME=~/cache/huggingface
export HF_HOME=/data/huggingface
export GPU_MEM_UTIL=0.4
mkdir -p data/corpus_construction/llm_extraction/${NEWSPAPER}
echo "===== NEWSPAPER: ${NEWSPAPER} | VARIANT: ${VARIANT} | LLM: ${LLM} ====="
python3 -m src.workflows.llm_extraction \
  --ocr-parquet data/corpus_construction/ocr_extraction/${NEWSPAPER}/${VARIANT}/ocr.parquet \
  --output-parquet data/corpus_construction/llm_extraction/${NEWSPAPER}/results_${VARIANT}_${LLM}.parquet \
  --llms ${LLM}