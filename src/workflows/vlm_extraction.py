"""
vlm_extraction.py — CLI entry point for the VLM text extraction step.

Runs one or more VLMs fully offline using vLLM's in-process LLM API.

The pipeline consumes the successful outputs recorded in the binarization
Parquet and extracts structured article metadata:

    - title
    - subheadline
    - author
    - body

The final output is one global Parquet file containing one row per:

    binarized image × VLM

Typical usage
-------------

Run all three VLMs:

    python -m src.workflows.vlm_extraction \
        --binarization-parquet \
        data/corpus_construction/binarization/results.parquet \
        --binarized-dir \
        data/corpus_construction/binarization \
        --output-parquet \
        data/corpus_construction/vlm_extraction/results.parquet

Run only RolmOCR:

    python -m src.workflows.vlm_extraction \
        --binarization-parquet \
        data/corpus_construction/binarization/results.parquet \
        --binarized-dir \
        data/corpus_construction/binarization \
        --vlms rolmocr

The VLMs are loaded one at a time so that GPU memory can be released
before loading the next model.
"""

import argparse
import logging
from pathlib import Path

from src.corpus_construction.vlm_extraction.pipeline import (
    VLMExtractionPipeline,
)

logger = logging.getLogger(__name__)

AVAILABLE_VLMS = [
    "olmocr",
    "rolmocr",
    "nanonets",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run offline VLM text extraction on binarized "
            "newspaper article images."
        )
    )

    parser.add_argument(
        "--binarization-parquet",
        required=True,
        help=(
            "Parquet produced by the binarization step. "
            "Only rows with status='success' are processed."
        ),
    )

    parser.add_argument(
        "--binarized-dir",
        required=True,
        help=(
            "Root directory containing the binarized TIFF files. "
            "Paths stored in binarize_file are resolved relative to "
            "this directory."
        ),
    )

    parser.add_argument(
        "--output-parquet",
        default=(
            "data/corpus_construction/"
            "vlm_extraction/results.parquet"
        ),
        help="Output Parquet path for VLM extraction results.",
    )

    parser.add_argument(
        "--vlms",
        nargs="+",
        choices=AVAILABLE_VLMS,
        default=AVAILABLE_VLMS,
        help=(
            "Which VLMs to run. Default: all. "
            "Models are loaded one at a time."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help=(
            "Maximum tokens to generate per image "
            "(default: 2048)."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of images per offline vLLM batch call "
            "(default: 16)."
        ),
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help=(
            "Fraction of GPU memory vLLM may reserve "
            "(default: 0.85)."
        ),
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help=(
            "Number of GPUs used to shard each model "
            "(default: 1)."
        ),
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help=(
            "Override the model maximum context length."
        ),
    )

    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model dtype (default: bfloat16).",
    )

    parser.add_argument(
        "--no-skip-failed",
        action="store_true",
        help=(
            "Re-run extractions that previously failed. "
            "Default behavior is to skip successful results "
            "but retry failed results."
        ),
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    engine_kwargs = {
        "gpu_memory_utilization": (
            args.gpu_memory_utilization
        ),
        "tensor_parallel_size": (
            args.tensor_parallel_size
        ),
        "dtype": args.dtype,
    }

    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = (
            args.max_model_len
        )

    vlm_engine_kwargs = {
        vlm: engine_kwargs
        for vlm in args.vlms
    }

    pipeline = VLMExtractionPipeline(
        logger=logger,
        vlms=args.vlms,
        binarization_parquet=(
            args.binarization_parquet
        ),
        binarized_dir=(
            args.binarized_dir
        ),
        parquet_path=(
            args.output_parquet
        ),
        max_new_tokens=(
            args.max_new_tokens
        ),
        batch_size=(
            args.batch_size
        ),
        skip_failed_extractions=(
            not args.no_skip_failed
        ),
        vlm_engine_kwargs=(
            vlm_engine_kwargs
        ),
    )

    try:
        processed = pipeline.run()

        logger.info(
            "VLM extraction finished. "
            "Extractions run: %d",
            processed,
        )

    except Exception as exc:
        logger.exception(
            "VLM extraction failed: %s",
            exc,
        )


if __name__ == "__main__":
    main()