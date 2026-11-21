#!/bin/bash
pkill -u $USER -f "ollama serve"
export OLLAMA_CONTEXT_LENGTH=49152
export PATH="$HOME/project/leonrios/tools/ollama/bin:$PATH"
export OLLAMA_HOST=127.0.0.1:11434                  # use another port if 11434 is taken
export OLLAMA_MODELS="$HOME/.ollama/models"      # weights location
export OLLAMA_CONTEXT_LENGTH=${CTX:-32768}      # per slot; override with: CTX=65536 source start_ollama.sh
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_MAX_LOADED_MODELS=3
export OLLAMA_KEEP_ALIVE=-1
export OLLAMA_FLASH_ATTENTION=1
export OLLAMA_KV_CACHE_TYPE=q8_0

nohup ollama serve > ./ollama_server.log 2>&1 &
sleep 3 && tail ./ollama_server.log

for m in mistral:7b qwen2.5:7b deepseek-r1:7b; do
  ollama pull "$m"
  curl -s http://$OLLAMA_HOST/api/generate -d "{\"model\":\"$m\",\"keep_alive\":-1}" > /dev/null
done

ollama ps          # all three should be listed with their GPU share
nvidia-smi         # check free memory on the shared A6000
nvidia-smi --query-gpu=memory.used,memory.total --format=csv

