"""Workflow: preprocess raw newspaper page images for OCR.

Each page is processed using all combinations of:

    Contrast (2):
        - None
        - CLAHE

    Denoising (9):
        - None
        - Mean Filter
        - Gaussian Filter
        - Median Filter
        - Conservative Filter
        - Laplacian Filter
        - Frequency Filtering
        - Crimmins Speckle Removal
        - Unsharp Filter

    Sharpening (3):
        - None
        - Unsharp Masking
        - Stroke-Width Enhancement

Total:

    2 x 9 x 3 = 54 preprocessing combinations

The processing order is:

    Contrast -> Denoising -> Sharpening

Usage:

    python3 -m src.workflows.enhance_images \
        --input-dir data/raw/images \
        --output-dir data/processed/enhanced \
        --newspaper el_comercio

The --newspaper argument is optional.
"""

import argparse
import logging
from pathlib import Path

from src.corpus_construction.enhance_images.pipeline import (
    EnhancementPipeline,
)


logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Apply all 54 image preprocessing "
            "combinations to raw newspaper images."
        )
    )

    parser.add_argument(
        "--input-dir",
        default="data/raw/images",
        help=(
            "Root directory containing "
            "downloaded images."
        ),
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/enhanced",
        help=(
            "Root directory where "
            "preprocessed images are saved."
        ),
    )

    parser.add_argument(
        "--newspaper",
        default=None,
        help=(
            "Process only this newspaper "
            "subdirectory. Defaults to all."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Re-process all images even if "
            "they were already processed."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s "
            "[%(levelname)s] "
            "%(name)s: %(message)s"
        ),
    )

    input_root = Path(
        args.input_dir
    )

    output_root = Path(
        args.output_dir
    )

    if args.newspaper:

        input_root = (
            input_root
            / args.newspaper
        )

        output_root = (
            output_root
            / args.newspaper
        )

    if not input_root.exists():

        logger.error(
            "Input directory does not exist: %s",
            input_root,
        )

        return

    pipeline = (
        EnhancementPipeline()
    )

    pipeline.run(
        input_root=input_root,
        output_root=output_root,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()