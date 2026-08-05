#!/bin/bash
#SBATCH --job-name=correo
#SBATCH --output=logs/slurm/correo_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500MB
#SBATCH --time=00:10:00

# set -euo pipefail

mkdir -p logs/slurm

####### 1. Enhancement #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/enhance_images/bin/activate
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/ --output-dir data/corpus_construction/enhance_images/results/ --newspaper correo
# ############# Specs (10 image): 1x1GB and 1h30min

####### 3. Binarization #######
module purge
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0
source venv/corpus_construction/binarize/bin/activate
python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/correo --output-dir data/corpus_construction/binarize/correo/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/correo --output-dir data/corpus_construction/binarize/correo/cropped/
############# Specs (10 image): 1x1GB and 5min

# ## 5. Evaluate
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/evaluate_extraction/bin/activate
# python3 -m src.workflows.evaluate_extraction --results data/corpus_construction/vlm_extraction/results/correo/test.parquet --gold data/corpus_construction/evaluate_extraction/correo.csv --output-csv data/corpus_construction/evaluate_extraction/results/correo/test.csv --enhance-parquet data/corpus_construction/enhance_images/results/correo/enhance_images.parquet --binarize-parquet data/corpus_construction/binarize/correo/none/binarization.parquet 