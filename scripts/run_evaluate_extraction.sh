#!/bin/bash
set -euo pipefail
NEWSPAPER=$1; LLM=$2
cd ~/project/leonrios/ma_up
source .venv/corpus_construction/evaluate_extraction/bin/activate
echo "===== NEWSPAPER: ${NEWSPAPER} | LLM: ${LLM} ====="
python3 -m src.workflows.evaluate_extraction \
    --results "data/corpus_construction/llm_extraction/${NEWSPAPER}/results_none_${LLM}.parquet" \
    --gold "data/corpus_construction/evaluate_extraction/${NEWSPAPER}.csv" \
    --output-csv "data/corpus_construction/evaluate_extraction/results/${NEWSPAPER}/${LLM}.csv" \
    --enhance-parquet "data/corpus_construction/enhance_images/results/${NEWSPAPER}/enhance_images.parquet" \
    --binarize-parquet "data/corpus_construction/binarize/${NEWSPAPER}/none/binarization.parquet" \
    --ocr-parquet "data/corpus_construction/ocr_extraction/${NEWSPAPER}/none/ocr.parquet"
python3 -m src.workflows.evaluate_extraction \
    --results "data/corpus_construction/llm_extraction/${NEWSPAPER}/results_cropped_${LLM}.parquet" \
    --gold "data/corpus_construction/evaluate_extraction/${NEWSPAPER}.csv" \
    --output-csv "data/corpus_construction/evaluate_extraction/results/${NEWSPAPER}/${LLM}.csv" \
    --enhance-parquet "data/corpus_construction/enhance_images/results/${NEWSPAPER}/enhance_images.parquet" \
    --binarize-parquet "data/corpus_construction/binarize/${NEWSPAPER}/cropped/binarization.parquet" \
    --ocr-parquet "data/corpus_construction/ocr_extraction/${NEWSPAPER}/cropped/ocr.parquet" \
    --layout-parquet "data/corpus_construction/layout_detection/${NEWSPAPER}/layout_detection.parquet"
