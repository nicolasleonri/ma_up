#!/bin/bash
#SBATCH --job-name=download_model
#SBATCH --output=logs/slurm/download_model_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=01:00:00


### 3. VLM
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
source venv/corpus_construction/vlm_extraction/bin/activate

hf cache list
# hf download allenai/olmOCR-2-7B-1025-FP8
# hf download AccsoAndreBuesgen/RolmOCR-bnb-4bit
# hf download sayed0am/Nanonets-OCR2-3B-FP8-Dynamic

