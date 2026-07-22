#!/bin/bash
#SBATCH --job-name=corpus_construction
#SBATCH --output=logs/slurm/corpus_construction_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=15GB
#SBATCH --time=01:00:00

# set -euo pipefail

mkdir -p logs/slurm

####### 1. Enhancement #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/enhance_images/bin/activate
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/test/ --output-dir data/corpus_construction/enhance_images/results/ --newspaper gestion
####### Specs (1 image): 1x200mB and 5min

####### 2. Layout #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/layout_detection/bin/activate
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/gestion --output-dir data/corpus_construction/layout_detection/
####### Specs (1 image): 1x2GB; 1xa5000 and 5min

####### 3. Binarization #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/binarize/bin/activate
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/gestion --output-dir data/corpus_construction/binarize/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/ --output-dir data/corpus_construction/binarize/cropped/
# # ####### Specs (1 image): 1x100mB and 5min

####### 4. VLM #######
module purge
module add virtualenv/20.23.1-GCCcore-12.3.0
module add Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
source venv/corpus_construction/vlm_extraction/bin/activate
python -m src.workflows.vlm_extraction \
    --binarized-dir data/corpus_construction/binarize/none \
    --binarization-parquet data/corpus_construction/binarize/none/binarization.parquet \
    --gpu-memory-utilization 0.85
# python -m src.workflows.vlm_extraction \
#     --binarized-dir data/corpus_construction/binarize/cropped \
#     --binarization-parquet data/corpus_construction/binarize/cropped/binarization.parquet \
#     --gpu-memory-utilization 0.85

# ## 4. Evaluate
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/evaluate_extraction/bin/activate
# python3 -m src.workflows.evaluate_extraction --results data/corpus_construction/vlm_extraction/results.parquet --gold data/corpus_construction/evaluate_extraction/gold_standard.csv --output-csv data/corpus_construction/evaluate_extraction/evaluation.csv