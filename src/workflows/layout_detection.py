import argparse
import logging
from pathlib import Path

from src.corpus_construction.layout_detection.pipeline import LayoutAnalysisPipeline

logger = logging.getLogger(__name__)

# AVAILABLE_DETECTORS = ["layoutparser", "doclayout_yolo"]
AVAILABLE_DETECTORS = ["layoutparser"]


def parse_args():
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
        "--detectors",
        nargs="+",
        choices=AVAILABLE_DETECTORS,
        default=AVAILABLE_DETECTORS,
        help="Which detectors to run (default: all).",
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

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pipeline = LayoutAnalysisPipeline(
        logger=logger,
        grid_rows=args.grid_rows,
        grid_cols=args.grid_cols,
        score_threshold=args.score_threshold,
        detectors=args.detectors,
    )

    try:
        processed = pipeline.run(
            preprocessed_dir=Path(args.preprocessed_dir),
            output_dir=Path(args.output_dir),
        )
        logger.info("Layout analysis finished. Images processed: %s", processed)
    except Exception as e:
        logger.exception("Layout analysis failed: %s", e)


if __name__ == "__main__":
    main()