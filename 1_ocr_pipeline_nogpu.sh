#!/bin/bash
#SBATCH --job-name=ocr_extraction_cpu
#SBATCH --output=logs/slurm/ocr_extraction_cpu_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=standard
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=02:00:00

# set -euo pipefail

mkdir -p logs/slurm

####### 1. Enhancement #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/enhance_images/bin/activate

# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/correo --output-dir data/corpus_construction/enhance_images/results/correo
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/elcomercio --output-dir data/corpus_construction/enhance_images/results/elcomercio
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/gestion --output-dir data/corpus_construction/enhance_images/results/gestion
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/ojo --output-dir data/corpus_construction/enhance_images/results/ojo
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/peru21 --output-dir data/corpus_construction/enhance_images/results/peru21
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/publimetro --output-dir data/corpus_construction/enhance_images/results/publimetro
# python3 -m src.workflows.enhance_images --input-dir data/corpus_construction/enhance_images/trome --output-dir data/corpus_construction/enhance_images/results/trome
# ############# Specs (10 image): 1x1GB and 1h30min

####### 3. Binarization #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/binarize/bin/activate

# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/correo --output-dir data/corpus_construction/binarize/correo/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/ojo --output-dir data/corpus_construction/binarize/ojo/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/elcomercio --output-dir data/corpus_construction/binarize/elcomercio/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/gestion --output-dir data/corpus_construction/binarize/gestion/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/peru21 --output-dir data/corpus_construction/binarize/peru21/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/publimetro --output-dir data/corpus_construction/binarize/publimetro/none/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/enhance_images/results/trome --output-dir data/corpus_construction/binarize/trome/none/

# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/correo --output-dir data/corpus_construction/binarize/correo/cropped/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/ojo --output-dir data/corpus_construction/binarize/ojo/cropped/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/elcomercio --output-dir data/corpus_construction/binarize/elcomercio/cropped/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/gestion --output-dir data/corpus_construction/binarize/gestion/cropped/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/peru21 --output-dir data/corpus_construction/binarize/peru21/cropped/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/publimetro --output-dir data/corpus_construction/binarize/publimetro/cropped/
# python3 -m src.workflows.binarize --input-dir data/corpus_construction/layout_detection/results/trome --output-dir data/corpus_construction/binarize/trome/cropped/
############# Specs (10 image): 1x1GB and 5min

####### 4. OCR-Extractor #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/ocr_extraction/bin/activate
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/correo/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/correo/none/ --output-dir data/corpus_construction/ocr_extraction/correo/none/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/ojo/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/ojo/none/ --output-dir data/corpus_construction/ocr_extraction/ojo/none/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/elcomercio/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/elcomercio/none/ --output-dir data/corpus_construction/ocr_extraction/elcomercio/none/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/gestion/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/gestion/none/ --output-dir data/corpus_construction/ocr_extraction/gestion/none/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/peru21/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/peru21/none/ --output-dir data/corpus_construction/ocr_extraction/peru21/none/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/publimetro/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/publimetro/none/ --output-dir data/corpus_construction/ocr_extraction/publimetro/none/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/trome/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/trome/none/ --output-dir data/corpus_construction/ocr_extraction/trome/none/

# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/correo/cropped/binarization.parquet --binarized-dir data/corpus_construction/binarize/correo/cropped/ --output-dir data/corpus_construction/ocr_extraction/correo/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/ojo/cropped/binarization.parquet --binarized-dir data/corpus_construction/binarize/ojo/cropped/ --output-dir data/corpus_construction/ocr_extraction/ojo/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/elcomercio/cropped/binarization.parquet --binarized-dir data/corpus_construction/binarize/elcomercio/cropped/ --output-dir data/corpus_construction/ocr_extraction/elcomercio/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/gestion/cropped/binarization.parquet --binarized-dir data/corpus_construction/binarize/gestion/cropped/ --output-dir data/corpus_construction/ocr_extraction/gestion/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/peru21/cropped/binarization.parquet --binarized-dir data/corpus_construction/binarize/peru21/cropped/ --output-dir data/corpus_construction/ocr_extraction/peru21/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/publimetro/cropped/binarization.parquet --binarized-dir data/corpus_construction/binarize/publimetro/cropped/ --output-dir data/corpus_construction/ocr_extraction/publimetro/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/trome/cropped/binarization.parquet --binarized-dir data/corpus_construction/binarize/trome/cropped/ --output-dir data/corpus_construction/ocr_extraction/trome/cropped/
############# Specs (1 image): TEST: 2x5GB and 60min

# ## 6. Evaluate
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/evaluate_extraction/bin/activate
# python3 -m src.workflows.evaluate_extraction --results data/corpus_construction/vlm_extraction/results/correo/test.parquet --gold data/corpus_construction/evaluate_extraction/correo.csv --output-csv data/corpus_construction/evaluate_extraction/results/correo/test.csv --enhance-parquet data/corpus_construction/enhance_images/results/correo/enhance_images.parquet --binarize-parquet data/corpus_construction/binarize/correo/none/binarization.parquet 