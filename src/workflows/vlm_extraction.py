"""
vlm_extraction.py — CLI entry point for the VLM text extraction step.

All VLMs run fully offline (in-process, via vLLM's offline LLM API) — no
vLLM server needs to be started separately, and no server URLs are needed.
This is intended for HPC compute nodes (e.g. `srun --pty bash` on curta)
where you can't open ports to a long-running server process.

Typical usage
-------------
# Run all three VLMs against all crops, batches of 16 crops per GPU call
python -m src.workflows.vlm_extraction \\
    --crops-dir  data/corpus_construction/layout_detection/crops \\
    --layout-parquet data/corpus_construction/layout_detection/results.parquet \\
    --batch-size 16

# Run only RolmOCR
python ... --vlms rolmocr

# Tune GPU memory usage / context length for a specific model
python ... --vlms olmocr --gpu-memory-utilization 0.7 --max-model-len 4096
"""

import argparse
import logging
from pathlib import Path

from src.corpus_construction.vlm_extraction.pipeline import VLMExtractionPipeline

logger = logging.getLogger(__name__)

AVAILABLE_VLMS = ["olmocr", "rolmocr", "nanonets"]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run offline VLM text extraction on newspaper article crops."
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
        help="Which VLMs to run (default: all). Models load one at a time, "
        "in this order, so they don't all need to fit on the GPU simultaneously.",
    )

    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=2048,
        help="Maximum tokens to generate per crop (default: 2048).",
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Number of crops per offline vLLM batch call (default: 16). "
        "Larger batches improve GPU throughput but use more memory.",
    )

    # --- vLLM engine tuning (applied to every requested VLM) ---
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.85,
        help="Fraction of GPU memory vLLM is allowed to reserve (default: 0.85).",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Number of GPUs to shard each model across (default: 1).",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=None,
        help="Override the model's max context length (default: model default).",
    )
    parser.add_argument(
        "--dtype",
        default="bfloat16",
        help="Model weight/activation dtype (default: bfloat16).",
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

    engine_kwargs = dict(
        gpu_memory_utilization=args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
        dtype=args.dtype,
    )
    if args.max_model_len is not None:
        engine_kwargs["max_model_len"] = args.max_model_len

    # Same engine settings applied to every requested VLM. Pass a dict here
    # instead if you need per-model overrides (e.g. {"olmocr": {...}}).
    vlm_engine_kwargs = {vlm: engine_kwargs for vlm in args.vlms}

    pipeline = VLMExtractionPipeline(
        logger=logger,
        vlms=args.vlms,
        layout_parquet=args.layout_parquet,
        parquet_path=args.output_parquet,
        max_new_tokens=args.max_new_tokens,
        batch_size=args.batch_size,
        skip_failed_crops=not args.no_skip_failed,
        vlm_engine_kwargs=vlm_engine_kwargs,
    )
    print(Path(args.crops_dir))
    try:
        processed = pipeline.run(crops_root=Path(args.crops_dir))
        logger.info("VLM extraction finished. Extractions run: %d", processed)
    except Exception as e:
        logger.exception("VLM extraction failed: %s", e)


if __name__ == "__main__":
    main()