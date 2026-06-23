#!/bin/bash
#SBATCH --job-name=corpus_construction
#SBATCH --output=logs/slurm/corpus_construction_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=00:30:00

# set -euo pipefail

mkdir -p logs/slurm

# ### 1. PP
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/preprocessing/bin/activate
# python3 -m src.workflows.preprocess_images --input-dir data/corpus_construction/preprocessing/test/ --output-dir data/corpus_construction/preprocessing/results/
# module purge

# ### 2. Layout
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/layout_detection/bin/activate
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/layout_detection/test/ --output-dir data/corpus_construction/layout_detection/results/ --detectors doclayout
# module purge

### 3. VLM
module purge
module add virtualenv/20.23.1-GCCcore-12.3.0
module add Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
source venv/corpus_construction/vlm_extraction/bin/activate
python3 -m src.workflows.vlm_extraction --crops-dir data/corpus_construction/layout_detection/results --gpu-memory-utilization 0.85

## 4. Evaluate
module purge
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0
source venv/corpus_construction/evaluate_extraction/bin/activate
python3 -m src.workflows.evaluate_extraction --results data/corpus_construction/vlm_extraction/results.parquet --gold data/corpus_construction/evaluate_extraction/gold_standard.csv --output-csv data/corpus_construction/evaluate_extraction/evaluation.csv
    