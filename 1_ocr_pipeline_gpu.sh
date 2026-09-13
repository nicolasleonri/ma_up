#!/bin/bash
#SBATCH --job-name=llm_extraction_all
#SBATCH --output=logs/slurm/llm_extraction_all_%j.out
#SBATCH --partition=scavenger
#SBATCH --account=agfritz
#SBATCH --qos=prio
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=5GB
#SBATCH --time=12:00:00

# set -euo pipefail

mkdir -p logs/slurm

####### 5. LLM Extractor #######
module purge
module add GCC/12.3.0
module add virtualenv/20.23.1-GCCcore-12.3.0
module add Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
module load cuDNN/8.9.2.26-CUDA-12.1.1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export HF_HOME=/scratch/nicolasal97/.cache/huggingface
source venv/corpus_construction/llm_extraction/bin/activate

for newspaper in correo elcomercio gestion ojo peru21 publimetro trome; do
# for newspaper in correo; do
    echo "===== NEWSPAPER: ${newspaper} ====="
    for llm in qwen mistral llama deepseek; do
        echo "===== LLM: ${llm} ====="
        python3 -m src.workflows.llm_extraction \
            --ocr-parquet data/corpus_construction/ocr_extraction/${newspaper}/none/ocr.parquet \
            --output-parquet data/corpus_construction/llm_extraction/${newspaper}/results_none_${llm}.parquet \
            --llms ${llm}
        echo ""
    done
done

for newspaper in correo elcomercio gestion ojo peru21 publimetro trome; do
# for newspaper in correo; do
    echo "===== NEWSPAPER: ${newspaper} ====="
    for llm in qwen mistral llama deepseek; do
        echo "===== LLM: ${llm} ====="
        python3 -m src.workflows.llm_extraction \
            --ocr-parquet data/corpus_construction/ocr_extraction/${newspaper}/cropped/ocr.parquet \
            --output-parquet data/corpus_construction/llm_extraction/${newspaper}/results_cropped_${llm}.parquet \
            --llms ${llm}
        echo ""
    done
done
############# Specs (1 image): 1x5GB; 1xh100 and 10min

# ################# DONE ###################
####### VLM #######
# module purge
# module add virtualenv/20.23.1-GCCcore-12.3.0
# module add Python/3.11.3-GCCcore-12.3.0
# module load CUDA/12.1.1
# module load cuDNN/8.9.2.26-CUDA-12.1.1
# export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/vlm_extraction/bin/activate
# python3 -m src.workflows.vlm_extraction \
#     --binarized-dir data/corpus_construction/binarize/correo/none \
#     --binarization-parquet data/corpus_construction/binarize/correo/none/binarization.parquet \
#     --gpu-memory-utilization 0.40 \
#     --output-parquet data/corpus_construction/vlm_extraction/results/correo/none.parquet
# # python3 -m src.workflows.vlm_extraction \
# #     --binarized-dir data/corpus_construction/binarize/correo/cropped \
# #     --binarization-parquet data/corpus_construction/binarize/correo/cropped/binarization.parquet \
# #     --gpu-memory-utilization 0.50 \
# #     --output-parquet data/corpus_construction/vlm_extraction/results/correo/cropped.parquet
# ############## Specs (1 image): 2x15GB; 1xh100 and 120min

####### OCR-Extractor #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/ocr_extraction/bin/activate
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/test_run/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/test_run/none/ --output-dir data/corpus_construction/ocr_extraction/test_run/none/
# ############# Specs (1 image): ?

####### 2. Layout #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# export HF_HOME=/scratch/nicolasal97/.cache/huggingface
# source venv/corpus_construction/layout_detection/bin/activate
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/correo --output-dir data/corpus_construction/layout_detection/correo
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/elcomercio --output-dir data/corpus_construction/layout_detection/elcomercio
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/gestion --output-dir data/corpus_construction/layout_detection/gestion
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/ojo --output-dir data/corpus_construction/layout_detection/ojo
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/peru21 --output-dir data/corpus_construction/layout_detection/peru21
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/publimetro --output-dir data/corpus_construction/layout_detection/publimetro
# python3 -m src.workflows.layout_detection --preprocessed-dir data/corpus_construction/enhance_images/results/trome --output-dir data/corpus_construction/layout_detection/trome
############# Specs (1 image): 2x2GB; 1xa5000 and 30min

####### 4. OCR-Extractor #######
# module purge
# module add virtualenv/20.32.0-GCCcore-14.3.0
# module add Python/3.13.5-GCCcore-14.3.0
# source venv/corpus_construction/ocr_extraction/bin/activate

# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/correo/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/correo/none/ --output-dir data/corpus_construction/ocr_extraction/correo/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/ojo/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/ojo/none/ --output-dir data/corpus_construction/ocr_extraction/ojo/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/elcomercio/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/elcomercio/none/ --output-dir data/corpus_construction/ocr_extraction/elcomercio/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/gestion/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/gestion/none/ --output-dir data/corpus_construction/ocr_extraction/gestion/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/peru21/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/peru21/none/ --output-dir data/corpus_construction/ocr_extraction/peru21/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/publimetro/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/publimetro/none/ --output-dir data/corpus_construction/ocr_extraction/publimetro/none_gpu/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/binarize/trome/none/binarization.parquet --binarized-dir data/corpus_construction/binarize/trome/none/ --output-dir data/corpus_construction/ocr_extraction/trome/none_gpu/

# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/correo/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/correo/cropped/ --output-dir data/corpus_construction/ocr_extraction/correo/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/ojo/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/ojo/cropped/ --output-dir data/corpus_construction/ocr_extraction/ojo/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/elcomercio/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/elcomercio/cropped/ --output-dir data/corpus_construction/ocr_extraction/elcomercio/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/gestion/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/gestion/cropped/ --output-dir data/corpus_construction/ocr_extraction/gestion/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/peru21/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/peru21/cropped/ --output-dir data/corpus_construction/ocr_extraction/peru21/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/publimetro/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/publimetro/cropped/ --output-dir data/corpus_construction/ocr_extraction/publimetro/cropped/
# python3 -m src.workflows.ocr_extraction --binarization-parquet data/corpus_construction/layout_detection/trome/cropped/binarization.parquet --binarized-dir data/corpus_construction/layout_detection/trome/cropped/ --output-dir data/corpus_construction/ocr_extraction/trome/cropped/
############# Specs (1 image): TEST: 2x15GB and 60min