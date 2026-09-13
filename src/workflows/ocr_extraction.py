"""CLI entry point for OCR extraction."""

import argparse
import logging
import os
from pathlib import Path

from src.corpus_construction.ocr_extraction.pipeline import (
    OCRExtractionPipeline,
)


AVAILABLE_OCR_EXTRACTORS = [
    # "docling",
    "docling_easyocr",
    "docling_rapidocr",
    # "docling_nemotron-ocr",
]


def parse_args():
    slurm_cpus = os.environ.get(
        "SLURM_CPUS_PER_TASK"
    )

    default_workers = int(slurm_cpus) if slurm_cpus else 1

    parser = argparse.ArgumentParser(
        description=(
            "Run OCR extractors over successful "
            "binarization outputs."
        )
    )

    parser.add_argument(
        "--binarization-parquet",
        required=True,
        help=(
            "Parquet file containing successful "
            "binarization outputs."
        ),
    )

    parser.add_argument(
        "--binarized-dir",
        required=True,
        help="Directory containing the binarized images.",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for OCR output.",
    )

    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=AVAILABLE_OCR_EXTRACTORS,
        default=AVAILABLE_OCR_EXTRACTORS,
        help="OCR extractors to run.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=(
            "Number of parallel OCR worker processes. "
            "Defaults to SLURM_CPUS_PER_TASK when available."
        ),
    )

    parser.add_argument(
        "--no-skip-failed",
        action="store_true",
        help="Reprocess previously failed OCR jobs.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    output_dir = Path(args.output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_parquet = output_dir / "ocr.parquet"

    pipeline = OCRExtractionPipeline(
        logger=logging.getLogger(__name__),
        extractors=args.extractors,
        binarization_parquet=args.binarization_parquet,
        binarized_dir=args.binarized_dir,
        parquet_path=output_parquet,
        skip_failed=not args.no_skip_failed,
        workers=args.workers,
    )

    n = pipeline.run()

    logging.info(
        "OCR extraction finished: %d results",
        n,
    )


if __name__ == "__main__":
    raise SystemExit(main())