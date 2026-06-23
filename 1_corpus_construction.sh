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

### 1. PP
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0
source venv/corpus_construction/preprocessing/bin/activate
python3 -m src.workflows.preprocess_images --input-dir data/corpus_construction/preprocessing/test/ --output-dir data/corpus_construction/preprocessing/results/
module purge

### 2. Layout
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
source venv/corpus_construction/layout_detection/bin/activate
python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/layout_detection/test/ --output-dir data/corpus_construction/layout_detection/results/ --detectors doclayout
module purge