import argparse
import logging
import sys
from pathlib import Path

from src.corpus_construction.binarize.pipeline import (
    BinarizationPipeline,
)


logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Apply all binarization methods to enhanced full-page "
            "and cropped article images."
        )
    )

    parser.add_argument(
        "--input-dir",
        type=Path,
        required=True,
        help=(
            "Directory containing enhanced input images. "
            "Images may be full-page or cropped."
        ),
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help=(
            "Directory where binarized images, metadata, "
            "configuration information, and checkpoints are stored."
        ),
    )

    parser.add_argument(
        "--layout-parquet",
        type=Path,
        default=None,
        help=(
            "Layout Parquet used to resolve image_stem and detector "
            "for cropped images."
        ),
    )

    parser.add_argument(
        "--no-resume",
        action="store_true",
        help=(
            "Ignore the checkpoint and process all input images again."
        ),
    )

    return parser.parse_args()


def main() -> int:
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format=(
            "%(asctime)s %(levelname)s "
            "%(name)s %(message)s"
        ),
    )

    pipeline = BinarizationPipeline(
        logger=logger,
        layout_parquet=args.layout_parquet,
    )

    try:
        processed = pipeline.run(
            input_root=args.input_dir,
            output_root=args.output_dir,
            resume=not args.no_resume,
        )

        logger.info(
            "Successfully processed %d images",
            processed,
        )

        return 0

    except Exception:
        logger.exception(
            "Binarization preprocessing failed"
        )
        return 1


if __name__ == "__main__":
    sys.exit(main())