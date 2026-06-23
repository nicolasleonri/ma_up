"""
vlm_extraction.py — CLI entry point for the VLM text extraction step.

Typical usage
-------------
# Run all three VLMs against all crops
python -m src.corpus_construction.vlm_extraction.vlm_extraction \\
    --crops-dir  data/corpus_construction/layout_detection/crops \\
    --layout-parquet data/corpus_construction/layout_detection/results.parquet

# Run only RolmOCR (fastest, no metadata needed)
python ... --vlms rolmocr

# Use OlmOCR locally (no vLLM server) with NanonetsOCR via vLLM
python ... --vlms olmocr nanonets \\
    --olmocr-local \\
    --nanonets-server http://gpu-node-3:8003/v1

vLLM server commands (run each in a separate terminal / tmux pane)
------------------------------------------------------------------
  OlmOCR    : vllm serve allenai/olmOCR-2-7B-1025 --port 8001
  RolmOCR   : VLLM_USE_V1=1 vllm serve reducto/RolmOCR --port 8002
  NanonetsOCR: vllm serve nanonets/Nanonets-OCR-s --port 8003
"""

import argparse
import logging
from pathlib import Path

from src.corpus_construction.vlm_extraction.vlm_pipeline import VLMExtractionPipeline

logger = logging.getLogger(__name__)

AVAILABLE_VLMS = ["olmocr", "rolmocr", "nanonets"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run VLM text extraction on newspaper article crops."
    )

    parser.add_argument(
        "--crops-dir",
        required=True,
        help="Root directory of article crop TIFFs (output of layout analysis).",
    )

    parser.add_argument(
        "--layout-parquet",
        default="data/corpus_construction/layout_detection/results.parquet",
        help="Layout analysis Parquet file that lists all crop files and metadata.",
    )

    parser.add_argument(
        "--output-parquet",
        default="data/corpus_construction/vlm_extraction/results.parquet",
        help="Output Parquet path for VLM extraction results.",
    )

    parser.add_argument(
        "--vlms",
        nargs="+",
        choices=AVAILABLE_VLMS,
        default=AVAILABLE_VLMS,
        help="Which VLMs to run (default: all).",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per crop (default: 2048).",
    )

    # --- Per-model server URLs ---
    parser.add_argument(
        "--olmocr-server",
        default="http://localhost:8001/v1",
        help="vLLM server URL for OlmOCR (default: http://localhost:8001/v1).",
    )
    parser.add_argument(
        "--rolmocr-server",
        default="http://localhost:8002/v1",
        help="vLLM server URL for RolmOCR (default: http://localhost:8002/v1).",
    )
    parser.add_argument(
        "--nanonets-server",
        default="http://localhost:8003/v1",
        help="vLLM server URL for NanonetsOCR (default: http://localhost:8003/v1).",
    )

    # --- Local-model flags ---
    parser.add_argument(
        "--olmocr-local",
        action="store_true",
        help="Load OlmOCR locally via Transformers instead of using a vLLM server.",
    )
    parser.add_argument(
        "--nanonets-local",
        action="store_true",
        help="Load NanonetsOCR locally via Transformers instead of using a vLLM server.",
    )

    parser.add_argument(
        "--no-skip-failed",
        action="store_true",
        help="Re-run extractions that previously failed (default: skip them).",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    pipeline = VLMExtractionPipeline(
        logger=logger,
        vlms=args.vlms,
        layout_parquet=args.layout_parquet,
        parquet_path=args.output_parquet,
        olmocr_server_url=args.olmocr_server,
        rolmocr_server_url=args.rolmocr_server,
        nanonets_server_url=args.nanonets_server,
        olmocr_use_local=args.olmocr_local,
        nanonets_use_local=args.nanonets_local,
        max_new_tokens=args.max_new_tokens,
        skip_failed_crops=not args.no_skip_failed,
    )

    try:
        processed = pipeline.run(crops_root=Path(args.crops_dir))
        logger.info("VLM extraction finished. Extractions run: %d", processed)
    except Exception as e:
        logger.exception("VLM extraction failed: %s", e)


if __name__ == "__main__":
    main()