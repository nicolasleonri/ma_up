#!/bin/bash
#SBATCH --job-name=corpus_acquisition
#SBATCH --output=logs/slurm/corpus_acquisition_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=2GB
#SBATCH --time=12:00:00

module purge
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0

source venv/corpus_acquisition/bin/activate

python3 -m src.workflows.acquire_corpus \
    --newspaper elcomercio \
    --start-date 2023-01-01 \
    --end-date 2023-12-31