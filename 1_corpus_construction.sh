#!/bin/bash
#SBATCH --job-name=corpus_construction
#SBATCH --output=logs/slurm/corpus_construction_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=01:00:00

set -euo pipefail

mkdir -p logs/slurm

module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0

source venv/corpus_construction/preprocessing/bin/activate
python3 -m src.workflows.preprocess_images --input-dir data/corpus_construction/test/ --output-dir data/corpus_construction/results/

####### LAYOUT

# module purge
# module add virtualenv/20.23.1-GCCcore-12.3.0
# module add Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# source venv/corpus_construction/layout_detection/layoutparser/bin/activate
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/layout_detection/test/ --output-dir data/corpus_construction/layout_detection/results/ --detectors layoutparser

####### DOCLAYOUT

module purge
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0

source venv/corpus_construction/layout_detection/doclayout/bin/activate
python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/layout_detection/test/ --output-dir data/corpus_construction/layout_detection/results/ --detectors doclayout

# SURYA
module purge
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0
# module add CMake/3.31.8-GCCcore-14.3.0
export PATH="$HOME/llama-server/build/bin:$PATH"
export SURYA_INFERENCE_BACKEND=llamacpp

source venv/corpus_construction/layout_detection/doclayout/bin/activate
python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/layout_detection/test/ --output-dir data/corpus_construction/layout_detection/results/ --detectors doclayout