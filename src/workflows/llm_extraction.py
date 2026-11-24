"""
llm_extraction.py — CLI entry point for local LLM structured extraction.
"""

import argparse
import logging
import os
from pathlib import Path

from src.corpus_construction.llm_extraction.pipeline import (
    DEFAULT_PROMPTS_DIR,
    LLMExtractionPipeline,
    NEWSPAPER_LLMS,
)

logger = logging.getLogger(__name__)

AVAILABLE_LLMS = [
    "qwen",
    "mistral",
    "llama",
    "deepseek",
]

FULL_PAGE_NEWSPAPERS = {"elcomercio", "gestion"}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run local in-process LLM structured extraction "
            "over OCR text with DSPy."
        )
    )

    parser.add_argument(
        "--newspaper",
        default=None,
        help="Newspaper key (correo, elcomercio, gestion, ojo, peru21, publimetro, trome).",
    )

    parser.add_argument(
        "--prompts-dir",
        default=str(DEFAULT_PROMPTS_DIR),
        help=f"Directory containing <newspaper>.md prompts (default: {DEFAULT_PROMPTS_DIR}).",
    )

    parser.add_argument(
        "--ocr-dir",
        default=None,
        help="Directory containing shard OCR Parquet files (ocr_*.parquet).",
    )

    parser.add_argument(
        "--ocr-parquet",
        default=None,
        help="Optional explicit single Parquet produced by the OCR extraction step.",
    )

    parser.add_argument(
        "--output-dir",
        default=None,
        help="Directory for shard-safe LLM extraction Parquet outputs.",
    )

    parser.add_argument(
        "--output-parquet",
        default=None,
        help="Optional explicit single output Parquet path.",
    )

    parser.add_argument(
        "--sub-dirs",
        nargs="+",
        default=None,
        help="Optional year shards (e.g., 2013 2014) for pueue jobs.",
    )

    parser.add_argument(
        "--llms",
        nargs="+",
        choices=AVAILABLE_LLMS,
        default=None,
        help="Which LLMs to run. Defaults to newspaper's winning LLM.",
    )

    parser.add_argument(
        "--all-llms",
        action="store_true",
        help="Run all 4 available LLMs even when --newspaper is specified.",
    )

    parser.add_argument(
        "--min-free-gib",
        type=float,
        default=36.0,
        help="Minimum free GPU memory (GiB) required before loading the next model (default: 36.0).",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per extraction (default: 4096).",
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=float(os.environ.get("GPU_MEM_UTIL", 0.85)),
        help="Fraction of GPU memory vLLM may reserve (default: 0.85).",
    )

    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs used to shard each model (default: 1).",
    )

    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override the model maximum context length.",
    )

    parser.add_argument(
        "--dtype",
        default="auto",
        help="Model dtype (default: auto).",
    )

    parser.add_argument(
        "--no-skip-failed",
        action="store_true",
        help="Re-run extractions that previously failed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    newspaper_key = args.newspaper.lower() if args.newspaper else None
    mode_subdir = (
        "none" if newspaper_key in FULL_PAGE_NEWSPAPERS else "cropped"
    )

    ocr_dir = args.ocr_dir
    if ocr_dir is None and args.ocr_parquet is None:
        if not newspaper_key:
            raise ValueError(
                "Provide --newspaper, --ocr-dir, or --ocr-parquet."
            )
        ocr_dir = str(
            Path("data/corpus_construction/ocr_extraction")
            / newspaper_key
            / mode_subdir
        )

    output_dir = args.output_dir
    if output_dir is None and args.output_parquet is None:
        if newspaper_key:
            output_dir = str(
                Path("data/corpus_construction/llm_extraction")
                / newspaper_key
                / mode_subdir
            )
        else:
            output_dir = "data/corpus_construction/llm_extraction"

    if args.all_llms:
        llms = AVAILABLE_LLMS
    elif args.llms is not None:
        llms = args.llms
    elif newspaper_key and newspaper_key in NEWSPAPER_LLMS:
        llms = NEWSPAPER_LLMS[newspaper_key]
        logger.info(
            "Using default winning LLM(s) for %s: %s",
            newspaper_key,
            llms,
        )
    else:
        llms = AVAILABLE_LLMS

    engine_kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
    }
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len

    llm_engine_kwargs = {llm: dict(engine_kwargs) for llm in llms}

    pipeline = LLMExtractionPipeline(
        logger=logger,
        llms=llms,
        ocr_parquet=args.ocr_parquet,
        ocr_dir=ocr_dir,
        parquet_path=args.output_parquet,
        output_dir=output_dir,
        sub_dirs=args.sub_dirs,
        newspaper=newspaper_key,
        prompts_dir=args.prompts_dir,
        max_new_tokens=args.max_new_tokens,
        min_free_gib=args.min_free_gib,
        skip_failed_extractions=not args.no_skip_failed,
        llm_engine_kwargs=llm_engine_kwargs,
    )

    try:
        processed = pipeline.run()
        logger.info(
            "LLM extraction finished. Extractions run: %d",
            processed,
        )
    except Exception as exc:
        logger.exception(
            "LLM extraction failed: %s",
            exc,
        )


if __name__ == "__main__":
    main()