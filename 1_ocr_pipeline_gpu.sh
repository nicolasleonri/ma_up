#!/bin/bash
#SBATCH --job-name=deepseek_extraction
#SBATCH --chdir=/work/leonrios/ma_up
#SBATCH --output=/work/leonrios/ma_up/logs/slurm/%x_%A_%a.out
#SBATCH --partition=gpu
#SBATCH --qos=normal
#SBATCH --gres=gpu:a100_40gb:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=10G
#SBATCH --time=24:00:00
#SBATCH --array=0-6%4

set -uo pipefail
module purge

NEWSPAPERS=(correo elcomercio gestion ojo peru21 publimetro trome)
NEWSPAPER=${NEWSPAPERS[$SLURM_ARRAY_TASK_ID]}
VARIANT=none
LLM=deepseek

export HF_HOME=/work/leonrios/hf
export HF_HUB_OFFLINE=1
export VLLM_USE_FLASHINFER_SAMPLER=0
export GPU_MEM_UTIL=0.9
source /work/leonrios/ma_up/.venv/corpus_construction/llm_extraction/bin/activate

echo "===== ${NEWSPAPER} | ${VARIANT} | ${LLM} | node $(hostname) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv

python3 -m src.workflows.llm_extraction \
  --ocr-parquet data/corpus_construction/ocr_extraction/${NEWSPAPER}/${VARIANT}/ocr.parquet \
  --output-parquet data/corpus_construction/llm_extraction/${NEWSPAPER}/results_${VARIANT}_${LLM}.parquet \
  --llms ${LLM} \
  --gpu-memory-utilization 0.9

############################################################################
################################## DONE ####################################
############################################################################
# mkdir -p logs/slurm

# ####### 5. LLM Extractor #######
# module purge
# module add GCC/12.3.0
# module add virtualenv/20.23.1-GCCcore-12.3.0
# module add Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/work/leonrios/hf

# /cache/leonrios/hf_models
# export VLLM_USE_FLASHINFER_SAMPLER=0
# source venv/corpus_construction/llm_extraction/bin/activate

# for newspaper in correo elcomercio gestion ojo peru21 publimetro trome; do
#     echo "===== NEWSPAPER: ${newspaper} ====="
#     for llm in deepseek; do
#     # for llm in qwen mistral llama deepseek; do
#         echo "===== LLM: ${llm} ====="
#         python3 -m src.workflows.llm_extraction \
#             --ocr-parquet data/corpus_construction/ocr_extraction/${newspaper}/cropped/ocr.parquet \
#             --output-parquet data/corpus_construction/llm_extraction/${newspaper}/results_cropped_${llm}.parquet \
#             --llms ${llm}
#         echo ""
#     done
# done

# # for newspaper in correo elcomercio gestion ojo peru21 publimetro trome; do
# for newspaper in elcomercio; do
#     echo "===== NEWSPAPER: ${newspaper} ====="
#     for llm in deepseek; do
#         echo "===== LLM: ${llm} ====="
#         python3 -m src.workflows.llm_extraction \
#             --ocr-parquet data/corpus_construction/ocr_extraction/${newspaper}/none/ocr.parquet \
#             --output-parquet data/corpus_construction/llm_extraction/${newspaper}/results_none_${llm}.parquet \
#             --llms ${llm}
#         echo ""
#     done
# done

# ############# Specs (1 image): 1x5GB; 1xh100 and 10min
# TIME per model per newspaper (min): 40min*4*2=320min (5,33h)
# TIME per model per newspaper (max): 80min*4*2=640min (10,66h)
# find -type f \( -iname "*.parquet" -o -iname "*.txt" -o -iname "*.csv" \) -exec du -h {} + | sort -hr

####### VLM #######
# module purge
# module add virtualenv/20.23.1-GCCcore-12.3.0
# module add Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/vlm_extraction/bin/activate
# python3 -m src.workflows.vlm_extraction \
#     --binarized-dir data/corpus_construction/binarize/correo/none \
#     --binarization-parquet data/corpus_construction/binarize/correo/none/binarization.parquet \
#     --gpu-memory-utilization 0.40 \
#     --output-parquet data/corpus_construction/vlm_extraction/results/correo/none.parquet
# # python3 -m src.workflows.vlm_extraction \
# #     --binarized-dir data/corpus_construction/binarize/correo/cropped \
# #     --binarization-parquet data/corpus_construction/binarize/correo/cropped/binarization.parquet \
# #     --gpu-memory-utilization 0.50 \
# #     --output-parquet data/corpus_construction/vlm_extraction/results/correo/cropped.parquet
# ############## Specs (1 image): 2x15GB; 1xh100 and 120min

####### OCR-Extractor #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/ocr_extraction/bin/activate
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/test_run/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/test_run/none/ --output-dir data/corpus_construction/ocr_extraction/test_run/none/
# ############# Specs (1 image): ?

# ####### 2. Layout #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/layout_detection/bin/activate
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/correo --output-dir data/corpus_construction/layout_detection/correo
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/elcomercio --output-dir data/corpus_construction/layout_detection/elcomercio
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/gestion --output-dir data/corpus_construction/layout_detection/gestion
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/ojo --output-dir data/corpus_construction/layout_detection/ojo
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/peru21 --output-dir data/corpus_construction/layout_detection/peru21
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/publimetro --output-dir data/corpus_construction/layout_detection/publimetro
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/trome --output-dir data/corpus_construction/layout_detection/trome
############# Specs (1 image): 2x2GB; 1xa5000 and 30min

####### 4. OCR-Extractor #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/ocr_extraction/bin/activate

# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/correo/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/correo/none/ --output-dir data/corpus_construction/ocr_extraction/correo/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/ojo/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/ojo/none/ --output-dir data/corpus_construction/ocr_extraction/ojo/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/elcomercio/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/elcomercio/none/ --output-dir data/corpus_construction/ocr_extraction/elcomercio/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/gestion/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/gestion/none/ --output-dir data/corpus_construction/ocr_extraction/gestion/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/peru21/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/peru21/none/ --output-dir data/corpus_construction/ocr_extraction/peru21/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/publimetro/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/publimetro/none/ --output-dir data/corpus_construction/ocr_extraction/publimetro/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/trome/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/trome/none/ --output-dir data/corpus_construction/ocr_extraction/trome/none_gpu/

# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/correo/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/correo/cropped/ --output-dir data/corpus_construction/ocr_extraction/correo/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/ojo/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/ojo/cropped/ --output-dir data/corpus_construction/ocr_extraction/ojo/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/elcomercio/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/elcomercio/cropped/ --output-dir data/corpus_construction/ocr_extraction/elcomercio/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/gestion/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/gestion/cropped/ --output-dir data/corpus_construction/ocr_extraction/gestion/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/peru21/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/peru21/cropped/ --output-dir data/corpus_construction/ocr_extraction/peru21/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/publimetro/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/publimetro/cropped/ --output-dir data/corpus_construction/ocr_extraction/publimetro/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/trome/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/trome/cropped/ --output-dir data/corpus_construction/ocr_extraction/trome/cropped/
############# Specs (1 image): TEST: 2x15GB and 60min