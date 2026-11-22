"""CLI entry point for OCR extraction."""

import argparse
import logging
import os
from pathlib import Path

from src.corpus_construction.ocr_extraction.pipeline import (
    NEWSPAPER_OCR_EXTRACTORS,
    OCRExtractionPipeline,
)

AVAILABLE_OCR_EXTRACTORS = [
    "docling",
    "docling_easyocr",
    "docling_rapidocr",
    "docling_nemotron-ocr",
]

FULL_PAGE_NEWSPAPERS = {"elcomercio", "gestion"}


def parse_args():
    slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
    default_workers = int(slurm_cpus) if slurm_cpus else 1

    parser = argparse.ArgumentParser(
        description="Run OCR extractors over successful binarization outputs."
    )

    parser.add_argument(
        "--newspaper",
        default=None,
        help="Newspaper key (correo, elcomercio, gestion, ojo, peru21, publimetro, trome).",
    )

    parser.add_argument(
        "--binarized-dir",
        type=Path,
        default=None,
        help="Directory containing binarized images and binarization*.parquet.",
    )

    parser.add_argument(
        "--binarization-parquet",
        type=Path,
        default=None,
        help="Optional explicit binarization Parquet file.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for OCR Parquet output.",
    )

    parser.add_argument(
        "--extractors",
        nargs="+",
        choices=AVAILABLE_OCR_EXTRACTORS,
        default=None,
        help="Explicit OCR extractor(s) to run. Defaults to newspaper's winning extractor.",
    )

    parser.add_argument(
        "--sub-dirs",
        nargs="+",
        default=None,
        help="Optional year shards (e.g., 2013 2014) for parallel pueue jobs.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help="Parallel worker processes (use 1 on GPU to avoid VRAM duplication).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="Docling batch size when running in-process (workers=1).",
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

    newspaper_key = args.newspaper.lower() if args.newspaper else None

    if args.binarized_dir is not None:
        binarized_dir = args.binarized_dir
    elif newspaper_key:
        binarized_dir = Path("data/processed/binarized") / newspaper_key
    else:
        raise ValueError("Either --newspaper or --binarized-dir must be provided.")

    if args.output_dir is not None:
        output_dir = args.output_dir
    elif newspaper_key:
        mode_subdir = (
            "none" if newspaper_key in FULL_PAGE_NEWSPAPERS else "cropped"
        )
        output_dir = (
            Path("data/corpus_construction/ocr_extraction")
            / newspaper_key
            / mode_subdir
        )
    else:
        raise ValueError("Either --newspaper or --output-dir must be provided.")

    if args.extractors is not None:
        extractors = args.extractors
    elif newspaper_key and newspaper_key in NEWSPAPER_OCR_EXTRACTORS:
        extractors = NEWSPAPER_OCR_EXTRACTORS[newspaper_key]
        logging.info(
            "Using default OCR extractor(s) for %s: %s",
            newspaper_key,
            extractors,
        )
    else:
        extractors = ["docling_easyocr", "docling_rapidocr"]

    output_dir.mkdir(parents=True, exist_ok=True)

    pipeline = OCRExtractionPipeline(
        logger=logging.getLogger(__name__),
        extractors=extractors,
        binarized_dir=binarized_dir,
        output_dir=output_dir,
        binarization_parquet=args.binarization_parquet,
        sub_dirs=args.sub_dirs,
        newspaper=newspaper_key,
        skip_failed=not args.no_skip_failed,
        workers=args.workers,
        batch_size=args.batch_size,
    )

    n = pipeline.run()
    logging.info("OCR extraction finished: %d results", n)


if __name__ == "__main__":
    raise SystemExit(main())