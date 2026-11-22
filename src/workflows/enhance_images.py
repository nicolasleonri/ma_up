"""Workflow: parallel preprocessing of raw newspaper page images for OCR."""

import argparse
import logging
import os
from pathlib import Path

from src.corpus_construction.enhance_images.pipeline import (
    EnhancementPipeline,
    NEWSPAPER_CONFIG_IDS,
)

logger = logging.getLogger(__name__)


def parse_args():
    default_workers = int(
        os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1
    )

    parser = argparse.ArgumentParser(
        description="Apply selected image preprocessing combinations in parallel."
    )

    parser.add_argument(
        "--input-dir",
        default="data/raw/images",
        help="Root directory containing downloaded images.",
    )

    parser.add_argument(
        "--output-dir",
        default="data/processed/enhanced",
        help="Root directory where preprocessed images are saved.",
    )

    parser.add_argument(
        "--newspaper",
        default=None,
        help="Newspaper key (correo, elcomercio, gestion, ojo, peru21, publimetro, trome).",
    )

    parser.add_argument(
        "--configs",
        nargs="+",
        type=int,
        default=None,
        help="Explicit config_id(s) in 0..53 to run. Overrides --newspaper default.",
    )

    parser.add_argument(
        "--sub-dirs",
        nargs="+",
        default=None,
        help="Optional subdirectories (e.g., years 2013 2014) to process for sharded pueue jobs.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Number of parallel CPU worker processes (default: {default_workers}).",
    )

    parser.add_argument(
        "--all-configs",
        action="store_true",
        help="Run all 54 configurations even when --newspaper is specified.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Re-process all images even if they were already processed.",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    input_root = Path(args.input_dir).expanduser()
    output_root = Path(args.output_dir).expanduser()

    newspaper_key = args.newspaper.lower() if args.newspaper else None

    if newspaper_key:
        if input_root.name.lower() != newspaper_key:
            input_root = input_root / newspaper_key
        if output_root.name.lower() != newspaper_key:
            output_root = output_root / newspaper_key

    if not input_root.exists():
        logger.error("Input directory does not exist: %s", input_root)
        return

    if args.all_configs:
        config_ids = None
    elif args.configs is not None:
        config_ids = args.configs
    elif newspaper_key == "trome":
        config_ids = None  # Handled per-page in EnhancementPipeline (7 for last page, 34 otherwise)
    elif newspaper_key and newspaper_key in NEWSPAPER_CONFIG_IDS:
        config_ids = NEWSPAPER_CONFIG_IDS[newspaper_key]
        logger.info(
            "Using default winning config_id(s) for %s: %s",
            newspaper_key,
            config_ids,
        )
    else:
        config_ids = None

    pipeline = EnhancementPipeline(workers=args.workers)

    pipeline.run(
        input_root=input_root,
        output_root=output_root,
        resume=not args.no_resume,
        config_ids=config_ids,
        sub_dirs=args.sub_dirs,
        newspaper=newspaper_key,
    )


if __name__ == "__main__":
    main()