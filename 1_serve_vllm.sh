#!/usr/bin/env bash
# serve_vlms.sh
#
# Start vLLM servers for OlmOCR, RolmOCR, and NanonetsOCR in the background,
# each on its own port.  Designed to run on a multi-GPU node; adjust
# --tensor-parallel-size and CUDA_VISIBLE_DEVICES to your hardware.
#
# Usage
# -----
#   bash serve_vlms.sh            # start all three
#   bash serve_vlms.sh olmocr     # start only OlmOCR
#   bash serve_vlms.sh stop       # kill all servers started by this script
#
# Requirements
# ------------
#   pip install vllm
#   HUGGING_FACE_HUB_TOKEN must be set (or models already cached locally)
#
# Hardware notes
# --------------
#   OlmOCR     7B  — fits in one A100 40 GB (bfloat16)
#   RolmOCR    7B  — fits in one A100 40 GB (bfloat16); use FP8 on 24 GB GPUs
#   NanonetsOCR 3B — fits in one A100 40 GB or even a 24 GB GPU easily
#
# For a single-GPU machine serving one model at a time, run only one server
# and point the pipeline's --*-server flags accordingly.

set -euo pipefail

# ── Configuration ────────────────────────────────────────────────────────────

OLMOCR_MODEL="allenai/olmOCR-2-7B-1025-FP8"
ROLMOCR_MODEL="AccsoAndreBuesgen/RolmOCR-bnb-4bit"
NANONETS_MODEL="sayed0am/Nanonets-OCR2-3B-FP8-Dynamic"

OLMOCR_PORT=8001
ROLMOCR_PORT=8002
NANONETS_PORT=8003

# GPU assignments — change to match your node (e.g. "0,1" for two GPUs)
OLMOCR_GPUS="0"
ROLMOCR_GPUS="1"
NANONETS_GPUS="2"

LOG_DIR="logs/vllm"
PID_DIR=".vllm_pids"
mkdir -p "$LOG_DIR" "$PID_DIR"

# ── Helpers ──────────────────────────────────────────────────────────────────

wait_for_server() {
    local port=$1
    local name=$2
    local max_wait=120   # seconds
    local interval=5
    local elapsed=0

    echo "  Waiting for $name on port $port..."
    while ! curl -sf "http://localhost:$port/health" > /dev/null 2>&1; do
        sleep $interval
        elapsed=$((elapsed + interval))
        if [[ $elapsed -ge $max_wait ]]; then
            echo "  ERROR: $name did not become healthy after ${max_wait}s. Check $LOG_DIR/${name}.log"
            exit 1
        fi
    done
    echo "  $name is up."
}

start_server() {
    local name=$1
    local model=$2
    local port=$3
    local gpus=$4
    shift 4
    local extra_args=("$@")   # any additional vllm flags

    local pid_file="$PID_DIR/${name}.pid"
    if [[ -f "$pid_file" ]] && kill -0 "$(cat "$pid_file")" 2>/dev/null; then
        echo "[$name] already running (pid $(cat "$pid_file")), skipping."
        return
    fi

    echo "[$name] Starting on GPU(s) $gpus, port $port..."
    CUDA_VISIBLE_DEVICES=$gpus \
    nohup python -m vllm.entrypoints.openai.api_server \
        --model "$model" \
        --port "$port" \
        --dtype bfloat16 \
        --max-model-len 8192 \
        --gpu-memory-utilization 0.90 \
        "${extra_args[@]}" \
        > "$LOG_DIR/${name}.log" 2>&1 &

    echo $! > "$pid_file"
    echo "[$name] PID $! — logs at $LOG_DIR/${name}.log"
}

stop_all() {
    echo "Stopping all VLM servers..."
    for pid_file in "$PID_DIR"/*.pid; do
        [[ -f "$pid_file" ]] || continue
        local pid
        pid=$(cat "$pid_file")
        local name
        name=$(basename "$pid_file" .pid)
        if kill -0 "$pid" 2>/dev/null; then
            echo "  Stopping $name (pid $pid)..."
            kill "$pid"
        else
            echo "  $name (pid $pid) is not running."
        fi
        rm -f "$pid_file"
    done
    echo "Done."
}

# ── Main ─────────────────────────────────────────────────────────────────────

TARGET="${1:-all}"

if [[ "$TARGET" == "stop" ]]; then
    stop_all
    exit 0
fi

if [[ "$TARGET" == "all" || "$TARGET" == "olmocr" ]]; then
    start_server "olmocr" "$OLMOCR_MODEL" "$OLMOCR_PORT" "$OLMOCR_GPUS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "rolmocr" ]]; then
    # RolmOCR requires the V1 engine
    VLLM_USE_V1=1 \
    start_server "rolmocr" "$ROLMOCR_MODEL" "$ROLMOCR_PORT" "$ROLMOCR_GPUS"
fi

if [[ "$TARGET" == "all" || "$TARGET" == "nanonets" ]]; then
    start_server "nanonets" "$NANONETS_MODEL" "$NANONETS_PORT" "$NANONETS_GPUS"
fi

# Wait for each requested server to pass its health check
if [[ "$TARGET" == "all" || "$TARGET" == "olmocr" ]];  then wait_for_server $OLMOCR_PORT  "olmocr";   fi
if [[ "$TARGET" == "all" || "$TARGET" == "rolmocr" ]]; then wait_for_server $ROLMOCR_PORT "rolmocr";  fi
if [[ "$TARGET" == "all" || "$TARGET" == "nanonets" ]]; then wait_for_server $NANONETS_PORT "nanonets"; fi

echo ""
echo "All requested servers are ready.  Run the extraction pipeline with:"
echo ""
echo "  python -m src.corpus_construction.vlm_extraction.vlm_extraction \\"
echo "      --crops-dir data/corpus_construction/layout_detection/crops \\"
echo "      --olmocr-server  http://localhost:${OLMOCR_PORT}/v1 \\"
echo "      --rolmocr-server http://localhost:${ROLMOCR_PORT}/v1 \\"
echo "      --nanonets-server http://localhost:${NANONETS_PORT}/v1"
echo ""
echo "To stop all servers:  bash serve_vlms.sh stop"