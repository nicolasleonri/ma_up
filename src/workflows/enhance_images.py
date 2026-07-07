"""Workflow: enhance raw newspaper page images before layout detection.

Applies all combinations of CLAHE / denoising / unsharp masking to every
image under ``--input-dir``, writing results to ``--output-dir`` while
preserving the source directory structure.

Usage:
    python3 -m src.workflows.enhance_images \
        --input-dir data/raw/images \
        --output-dir data/processed/enhanced \
        --newspaper el_comercio          # optional: process one newspaper only
"""

import argparse
import logging
from pathlib import Path

from src.corpus_construction.enhance_images.pipeline import EnhancementPipeline

logger = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Enhance raw newspaper page images.")
    parser.add_argument(
        "--input-dir",
        default="data/raw/images",
        help="Root directory of downloaded images.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/processed/enhanced",
        help="Root directory for enhanced images.",
    )
    parser.add_argument(
        "--newspaper",
        default=None,
        help="Process only this newspaper subdirectory. Defaults to all.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process all images even if already enhanced.",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_root = Path(args.input_dir)
    output_root = Path(args.output_dir)

    if args.newspaper:
        input_root = input_root / args.newspaper
        output_root = output_root / args.newspaper

    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)
        return

    pipeline = EnhancementPipeline()
    pipeline.run(
        input_root=input_root,
        output_root=output_root,
        resume=not args.no_resume,
    )


if __name__ == "__main__":
    main()