import argparse
import logging
import os
from pathlib import Path

from src.corpus_construction.layout_detection.pipeline import (
    LayoutAnalysisPipeline,
    NEWSPAPER_DETECTORS,
)

logger = logging.getLogger(__name__)

AVAILABLE_DETECTORS = ["doclayout", "ppdoclayout", "histogram"]


def parse_args():
    default_workers = int(
        os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1
    )

    parser = argparse.ArgumentParser(
        description="Run layout analysis on preprocessed newspaper images."
    )

    parser.add_argument(
        "--preprocessed-dir",
        required=True,
        help="Directory containing preprocessed variants (*_config_N.tiff).",
    )

    parser.add_argument(
        "--output-dir",
        required=True,
        help="Root directory for layout analysis outputs.",
    )

    parser.add_argument(
        "--newspaper",
        default=None,
        help="Optional newspaper key to auto-select its winning detector(s).",
    )

    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=AVAILABLE_DETECTORS,
        default=None,
        help="Explicit detector(s) to run (e.g., --detectors doclayout ppdoclayout).",
    )

    parser.add_argument(
        "--sub-dirs",
        nargs="+",
        default=None,
        help="Optional year subdirectories (e.g., 2013 2014) for sharded pueue jobs.",
    )

    parser.add_argument(
        "--workers",
        type=int,
        default=default_workers,
        help=f"Parallel CPU workers for histogram detector (default: {default_workers}).",
    )

    parser.add_argument(
        "--grid-rows",
        type=int,
        default=3,
        help="Number of rows in the composition grid (default: 3).",
    )

    parser.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="Number of columns in the composition grid (default: 3).",
    )

    parser.add_argument(
        "--score-threshold",
        type=float,
        default=0.5,
        help="Minimum confidence score for a detection (default: 0.5).",
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
        format="%(asctime)s %(levelname)s %(message)s",
    )

    newspaper_key = args.newspaper.lower() if args.newspaper else None

    if args.detectors is not None:
        detectors = args.detectors
    elif newspaper_key and newspaper_key in NEWSPAPER_DETECTORS:
        detectors = NEWSPAPER_DETECTORS[newspaper_key]
        logger.info(
            "Using default detector(s) for %s: %s", newspaper_key, detectors
        )
    else:
        detectors = AVAILABLE_DETECTORS

    pipeline = LayoutAnalysisPipeline(
        logger=logger,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        score_threshold=args.score_threshold,
        detectors=detectors,
        workers=args.workers,
        newspaper=newspaper_key,
    )

    try:
        processed = pipeline.run(
            preprocessed_dir=Path(args.preprocessed_dir),
            output_dir=Path(args.output_dir),
            resume=not args.no_resume,
            sub_dirs=args.sub_dirs,
            newspaper=newspaper_key,
        )
        logger.info("Layout analysis finished. Images processed: %s", processed)
    except Exception as e:
        logger.exception("Layout analysis failed: %s", e)
        raise SystemExit(1)


if __name__ == "__main__":
    main()