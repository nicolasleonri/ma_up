import argparse
import logging
import os
import sys
from pathlib import Path

from src.corpus_construction.binarize.pipeline import (
    BINARIZATION_METHODS,
    BinarizationPipeline,
    FULL_PAGE_NEWSPAPERS,
    NEWSPAPER_BINARIZATION,
)

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    default_workers = int(
        os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1
    )

    parser = argparse.ArgumentParser(
        description="Apply binarization methods to enhanced or cropped newspaper images."
    )

    parser.add_argument(
        "--newspaper",
        default=None,
        help="Newspaper key (correo, elcomercio, gestion, ojo, peru21, publimetro, trome).",
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        default=None,
        help=(
            "Optional input directory override. Defaults to data/processed/enhanced/<newspaper> "
            "for full-page newspapers and data/processed/layout_detection/<newspaper> for cropped."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory override. Defaults to data/processed/binarized/<newspaper>.",
    )

    parser.add_argument(
        "--methods",
        nargs="+",
        choices=BINARIZATION_METHODS,
        default=None,
        help="Explicit binarization method(s) to run. Defaults to newspaper's winning method.",
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
        help=f"Number of parallel CPU worker processes (default: {default_workers}).",
    )

    parser.add_argument(
        "--all-methods",
        action="store_true",
        help="Run all 8 binarization methods even when --newspaper is specified.",
    )

    parser.add_argument(
        "--layout-parquet",
        type=Path,
        default=None,
        help="Optional layout Parquet path.",
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore the checkpoint and process all input images again.",
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )

    newspaper_key = args.newspaper.lower() if args.newspaper else None

    if args.input_dir is not None:
        input_root = args.input_dir
    elif newspaper_key:
        base = (
            Path("data/processed/enhanced")
            if newspaper_key in FULL_PAGE_NEWSPAPERS
            else Path("data/processed/layout_detection")
        )
        input_root = base / newspaper_key
    else:
        raise ValueError("Either --newspaper or --input-dir must be provided.")

    if args.output_dir is not None:
        output_root = args.output_dir
    elif newspaper_key:
        output_root = Path("data/processed/binarized") / newspaper_key
    else:
        raise ValueError("Either --newspaper or --output-dir must be provided.")

    if args.all_methods:
        methods = BINARIZATION_METHODS
    elif args.methods is not None:
        methods = args.methods
    elif newspaper_key and newspaper_key in NEWSPAPER_BINARIZATION:
        methods = NEWSPAPER_BINARIZATION[newspaper_key]
        logger.info(
            "Using default winning binarization method(s) for %s: %s",
            newspaper_key,
            methods,
        )
    else:
        methods = BINARIZATION_METHODS

    pipeline = BinarizationPipeline(
        logger=logger,
        layout_parquet=args.layout_parquet,
        methods=methods,
        workers=args.workers,
        newspaper=newspaper_key,
    )

    try:
        processed = pipeline.run(
            input_root=input_root,
            output_root=output_root,
            resume=not args.no_resume,
            sub_dirs=args.sub_dirs,
            newspaper=newspaper_key,
        )
        logger.info("Successfully processed %d images", processed)
        return 0
    except Exception:
        logger.exception("Binarization preprocessing failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())