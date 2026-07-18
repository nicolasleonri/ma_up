#!/bin/bash
#SBATCH --job-name=corpus_acquisition
#SBATCH --output=logs/slurm/corpus_acquisition_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=3GB
#SBATCH --time=48:00:00

module purge
module add virtualenv/20.32.0-GCCcore-14.3.0
module add Python/3.13.5-GCCcore-14.3.0

source venv/corpus_acquisition/bin/activate

##### EL COMERCIO #####
# python3 -m src.workflows.acquire_corpus \
#     --newspaper elcomercio \
#     --start-date 2018-05-18 \
#     --end-date 2019-12-31 # Final date

##### TODO LO DEMAS #####
# newspaper = trome, elcomercio, peru21, gestion, ojo, correo
python3 -m src.workflows.acquire_corpus \
    --newspaper correo \
    --start-date 2024-08-29 \
    --end-date 2026-06-30 # Final date

exit_code=$?

if [ "$exit_code" -eq 76 ]; then
    echo "Browser crashed. Resubmitting immediately with a fresh session..."
    sbatch "$0"
    exit 0
fi

if [ "$exit_code" -eq 75 ]; then
    echo "Rate-limited (403). Resubmitting in 40 minutes..."
    sbatch --begin=now+40minutes "$0"
    exit 0
fi

exit $exit_code

# sbatch -d afterany:26016299 0_corpus_acquisition.sh
