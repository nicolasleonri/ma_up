#!/bin/bash
#SBATCH --job-name=correo
#SBATCH --output=logs/slurm/correo_h100_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=15GB
#SBATCH --time=00:10:00

# set -euo pipefail

mkdir -p logs/slurm

####### 2. Layout #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/ocr_extraction/bin/activate
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/correo --output-dir data/corpus_construction/layout_detection/results/correo
############# Specs (1 image): 2x2GB; 1xa5000 and 30min

####### 4. OCR-Extractor #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/ocr_extraction/bin/activate
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/test_run/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/test_run/none/ --output-dir data/corpus_construction/ocr_extraction/test_run/none/
# ############# Specs (1 image): 

####### 5. LLM Extractor #######
module purge
module add GCC/12.3.0
module add virtualenv/20.23.1-GCCcore-12.3.0
module add Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
source venv/corpus_construction/llm_extraction/bin/activate

echo "===== COMPILERS ====="
which gcc
gcc --version
which g++
g++ --version
which nvcc
nvcc --version

echo "===== PYTHON ====="
which python3
python3 --version
pip freeze

echo "===== PACKAGES ====="
python3 -c "import torch; print('torch', torch.__version__, 'CUDA', torch.version.cuda)"
python3 -c "import vllm; print('vllm', vllm.__version__)"
python3 -c "import flashinfer; print('flashinfer', flashinfer.__version__)"

echo "===== OUTPUT ====="
python3 -m src.workflows.llm_extraction \
    --ocr-parquet data/corpus_construction/ocr_extraction/test_run/none/ocr.parquet \
    --output-parquet data/corpus_construction/llm_extraction/test_run/none/results.parquet
############# Specs (1 image): 2x2GB; 1xa5000 and 30min

# ####### 4. VLM #######
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