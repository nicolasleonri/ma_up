"""
vlm_extraction.py — CLI entry point for local VLM text extraction.

Runs one or more VLMs fully offline using vLLM's in-process LLM API.

DSPy is used as the structured extraction layer. No DSPy optimization
or fine-tuning is performed yet.

Architecture:

    image
        ↓
    local VLM / vLLM
        ↓
    DSPy structured extraction
        ↓
    title
    subheadline
    author
    body

No VLM server is required.

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

from src.corpus_construction.vlm_extraction.pipeline import (
    VLMExtractionPipeline,
)


logger = logging.getLogger(__name__)


AVAILABLE_VLMS = [
    "olmocr",
    "deepseek",
    "nanonets",
]


def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Run local in-process VLM extraction "
            "with DSPy structured output."
        )
    )

    parser.add_argument(
        "--binarization-parquet",
        required=True,
        help=(
            "Parquet produced by the binarization step. "
            "Only rows with status='ok' are processed."
        ),
    )

    parser.add_argument(
        "--binarized-dir",
        required=True,
        help=(
            "Root directory containing the binarized TIFF files. "
            "Paths stored in binarize_file are resolved relative "
            "to this directory."
        ),
    )

    parser.add_argument(
        "--output-parquet",
        default=(
            "data/corpus_construction/"
            "vlm_extraction/results.parquet"
        ),
        help=(
            "Output Parquet path for VLM extraction results."
        ),
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
        default=4096,
        help=(
            "Maximum tokens to generate per image "
            "(default: 4096)."
        ),
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help=(
            "Number of images per local VLM batch "
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
        help=(
            "Model dtype (default: bfloat16)."
        ),
    )

    parser.add_argument(
        "--no-skip-failed",
        action="store_true",
        help=(
            "Re-run extractions that previously failed. "
            "Default behavior is to skip failed results."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "%(levelname)s "
            "%(message)s"
        ),
    )

    # --------------------------------------------------------------
    # Engine configuration.
    #
    # These parameters are passed directly to the local vLLM LLM
    # constructor inside steps.py.
    #
    # There is intentionally NO server_url and NO HTTP endpoint.
    # --------------------------------------------------------------

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

        engine_kwargs[
            "max_model_len"
        ] = args.max_model_len

    # Each VLM gets the same local engine configuration.
    #
    # This remains configurable per model because later we may want
    # different settings for RolmOCR, OLMOCR, and Nanonets.
    vlm_engine_kwargs = {
        vlm: dict(engine_kwargs)
        for vlm in args.vlms
    }

    # --------------------------------------------------------------
    # Create pipeline.
    # --------------------------------------------------------------

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

    # --------------------------------------------------------------
    # Run extraction.
    # --------------------------------------------------------------

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