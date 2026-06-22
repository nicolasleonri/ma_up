import argparse
import logging
from pathlib import Path

from src.corpus_construction.preprocessing.pipeline import (
    ImagePreprocessingPipeline
)

logger=logging.getLogger(__name__)

def parse_args():
    parser=argparse.ArgumentParser()

    parser.add_argument(
        "--input-dir",
        required=True
    )

    parser.add_argument(
        "--output-dir",
        required=True
    )

    return parser.parse_args()

def main():
    args=parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s"
    )

    pipeline=ImagePreprocessingPipeline(logger)

    try:
        processed=pipeline.run(Path(args.input_dir), Path(args.output_dir))

        logger.info("Processed %s images", processed)
    except Exception as e:
        logger.exception(f"Preprocessing failed: {e}")

if __name__=="__main__":
    main()