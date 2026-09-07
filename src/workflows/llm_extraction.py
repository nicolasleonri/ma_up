"""
llm_extraction.py — CLI entry point for local LLM structured extraction.

Runs one or more text-only LLMs fully offline using vLLM's in-process
API. DSPy is used as the structured extraction layer, wired to a
proper dspy.LM subclass backed by the in-process vLLM model.

Architecture:

    OCR parquet (text / markdown column)
        ↓
    local LLM / vLLM (text-only, in-process)
        ↓
    DSPy structured extraction
        ↓
    title / subheadline / author / body

No vLLM server is required.

Typical usage
-------------

Run all configured LLMs:

    python -m src.workflows.llm_extraction \\
        --ocr-parquet \\
        data/corpus_construction/ocr_extraction/results.parquet \\
        --output-parquet \\
        data/corpus_construction/llm_extraction/results.parquet

Run only Qwen:

    python -m src.workflows.llm_extraction \\
        --ocr-parquet \\
        data/corpus_construction/ocr_extraction/results.parquet \\
        --llms qwen

The LLMs are loaded one at a time so that GPU memory can be released
before loading the next model.
"""

import argparse
import logging

from src.corpus_construction.llm_extraction.pipeline import (
    LLMExtractionPipeline,
)

logger = logging.getLogger(__name__)

AVAILABLE_LLMS = [
    "qwen",
    "mistral",
    "llama",
    "deepseek",
]


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run local in-process LLM structured extraction "
            "over OCR text with DSPy."
        )
    )

    parser.add_argument(
        "--ocr-parquet",
        required=True,
        help=(
            "Parquet produced by the OCR extraction step. "
            "Only rows with status='success' are processed."
        ),
    )

    parser.add_argument(
        "--output-parquet",
        default=(
            "data/corpus_construction/"
            "llm_extraction/results.parquet"
        ),
        help="Output Parquet path for LLM extraction results.",
    )

    parser.add_argument(
        "--llms",
        nargs="+",
        choices=AVAILABLE_LLMS,
        default=AVAILABLE_LLMS,
        help=(
            "Which LLMs to run. Default: all. "
            "Models are loaded one at a time."
        ),
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=4096,
        help="Maximum tokens to generate per extraction (default: 4096).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help=(
            "Number of OCR texts per LLM batch (default: 32). "
            "Text-only batches can be larger than VLM batches."
        ),
    )

    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.90,
        help="Fraction of GPU memory vLLM may reserve (default: 0.90).",
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
        help=(
            "Model dtype (default: auto). "
            "FP8 quantized models carry their own quantization config "
            "and should use 'auto' so vLLM reads it correctly."
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
        format="%(asctime)s %(levelname)s %(message)s",
    )

    engine_kwargs = {
        "gpu_memory_utilization": args.gpu_memory_utilization,
        "tensor_parallel_size": args.tensor_parallel_size,
        "dtype": args.dtype,
    }

    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len

    # Each LLM gets the same engine configuration.
    # Kept per-model so settings can diverge later if needed
    # (e.g. different context lengths for Mistral vs Llama).
    llm_engine_kwargs = {
        llm: dict(engine_kwargs)
        for llm in args.llms
    }

    pipeline = LLMExtractionPipeline(
        logger=logger,
        llms=args.llms,
        ocr_parquet=args.ocr_parquet,
        parquet_path=args.output_parquet,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
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