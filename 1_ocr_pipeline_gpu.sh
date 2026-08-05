#!/bin/bash
#SBATCH --job-name=corpus_construction
#SBATCH --output=logs/slurm/test_run_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=hiprio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=20GB
#SBATCH --time=02:00:00

# set -euo pipefail

mkdir -p logs/slurm

## #SBATCH --gres=gpu:h100:1

####### 1. Enhancement #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/enhance_images/bin/activate
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/test/ --output-dir data/corpus_construction/enhance_images/results/ --newspaper gestion
# ############# Specs (1 image): 1x200mB and 5min

####### 2. Layout #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/layout_detection/bin/activate
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/gestion --output-dir data/corpus_construction/layout_detection/results/gestion
# ############# Specs (1 image): 1x2GB; 1xa5000 and 5min

####### 3. Binarization #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/binarize/bin/activate
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/gestion --output-dir data/corpus_construction/binarize/gestion/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/gestion --output-dir data/corpus_construction/binarize/gestion/cropped/
# ############## Specs (1 image): 1x200mB and 1min 

####### 4. VLM #######
module purge
module add virtualenv/20.23.1-GCCcore-12.3.0
module add Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
source venv/corpus_construction/vlm_extraction/bin/activate
python3 -m src.workflows.vlm_extraction \
    --binarized-dir data/corpus_construction/binarize/gestion/none \
    --binarization-parquet data/corpus_construction/binarize/gestion/none/binarization.parquet \
    --gpu-memory-utilization 0.40 \
    --output-parquet data/corpus_construction/vlm_extraction/results/gestion/none.parquet
python3 -m src.workflows.vlm_extraction \
    --binarized-dir data/corpus_construction/binarize/gestion/cropped \
    --binarization-parquet data/corpus_construction/binarize/gestion/cropped/binarization.parquet \
    --gpu-memory-utilization 0.50 \
    --output-parquet data/corpus_construction/vlm_extraction/results/gestion/cropped.parquet
############## Specs (1 image): 2x15GB; 1xh100 and 120min

# ## 4. Evaluate
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/evaluate_extraction/bin/activate
# python3 -m src.workflows.evaluate_extraction --results data/corpus_construction/vlm_extraction/results/gestion/test.parquet --gold data/corpus_construction/evaluate_extraction/gestion.csv --output-csv data/corpus_construction/evaluate_extraction/results/gestion/test.csv --enhance-parquet data/corpus_construction/enhance_images/results/gestion/enhance_images.parquet --binarize-parquet data/corpus_construction/binarize/gestion/none/binarization.parquet 