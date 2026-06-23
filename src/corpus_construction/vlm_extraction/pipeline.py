"""
VLM extraction pipeline.

Reads article crop TIFFs produced by the layout analysis step and runs one
or more VLMs over each crop to extract title and body text.

Crops are grouped per VLM and pushed through `extractor.extract_batch()` in
chunks of `batch_size`, so vLLM's offline continuous batching actually gets
a real batch to schedule instead of one crop at a time.

Output schema (one row per crop × VLM)
---------------------------------------
newspaper          str     e.g. "elcomercio"
date               str     e.g. "2026-01-03"
page               str     e.g. "2"
image_stem         str     e.g. "elcomercio_2026-01-03_2"
preprocessing_config int
detector           str     layout detector that produced the crop
article_idx        int     1-based article index (top-to-bottom)
crop_file          str     filename of the source crop TIFF
vlm                str     "olmocr" | "rolmocr" | "nanonets"
title              str     extracted article headline
body               str     extracted article body text
raw_text           str     verbatim model output
elapsed_s          float   wall-clock seconds for this call
status             str     "ok" | "failed"
error              str     error message if status == "failed", else null
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import pandas as pd

from .steps import (
    ExtractionResult,
    OlmOCRExtractor,
    RolmOCRExtractor,
    NanonetsOCRExtractor,
    VLM_EXTRACTORS,
)

DEFAULT_PARQUET = "data/corpus_construction/vlm_extraction/results.parquet"

# Key columns that uniquely identify one extraction result
_KEY_COLS = [
    "image_stem",
    "preprocessing_config",
    "detector",
    "article_idx",
    "vlm",
]


class VLMExtractionPipeline:
    """
    Runs one or more offline VLM extractors over every article crop
    produced by the layout analysis step.

    Usage
    -----
    The pipeline reads a layout Parquet file (produced by LayoutAnalysisPipeline)
    to discover which crop TIFFs exist and what metadata to attach. For each
    requested VLM it collects every crop that hasn't been processed yet,
    chunks them into batches of `batch_size`, and calls
    `extractor.extract_batch(images, metadata_list)` once per chunk — vLLM
    handles continuous batching across the chunk on the GPU. Results are
    flushed to Parquet after every chunk so the pipeline is safe to re-run
    after interruption.

    Parameters
    ----------
    logger                : Python logger instance.
    vlms                  : List of VLM names to run (subset of VLM_EXTRACTORS).
    layout_parquet        : Path to the layout analysis results Parquet file.
    parquet_path          : Output Parquet path for VLM results.
    max_new_tokens        : Passed to every VLM.
    batch_size            : Number of crops per offline vLLM batch call.
    skip_failed_crops     : If True, skip crops that previously failed.
    vlm_engine_kwargs     : Optional dict of {vlm_name: {kwarg: value}} passed
                            straight through to that VLM's extractor
                            constructor (e.g. gpu_memory_utilization,
                            tensor_parallel_size, max_model_len, dtype).
    """

    def __init__(
        self,
        logger: logging.Logger,
        vlms: List[str] = None,
        layout_parquet: str = "data/corpus_construction/layout_detection/results.parquet",
        parquet_path: str = DEFAULT_PARQUET,
        max_new_tokens: int = 2048,
        batch_size: int = 16,
        skip_failed_crops: bool = True,
        vlm_engine_kwargs: Dict[str, Dict[str, Any]] = None,
        # Kept for backward compat with old CLI args; accepted and ignored
        # since every extractor is offline-only now.
        olmocr_server_url: str = None,
        rolmocr_server_url: str = None,
        nanonets_server_url: str = None,
        olmocr_use_local: bool = True,
        nanonets_use_local: bool = True,
    ):
        self.logger = logger
        self.vlm_names = vlms or list(VLM_EXTRACTORS.keys())
        self.layout_parquet = Path(layout_parquet)
        self.parquet_path = Path(parquet_path)
        self.max_new_tokens = max_new_tokens
        self.batch_size = batch_size
        self.skip_failed_crops = skip_failed_crops
        self.vlm_engine_kwargs = vlm_engine_kwargs or {}

        self._loaded_vlms: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # VLM cache
    # ------------------------------------------------------------------

    def _get_vlm(self, name: str):
        if name not in self._loaded_vlms:
            if name not in VLM_EXTRACTORS:
                raise ValueError(f"Unknown VLM: {name!r}. Choose from {list(VLM_EXTRACTORS)}")
            self.logger.info("Loading VLM extractor (offline, in-process): %s", name)
            extractor_cls = VLM_EXTRACTORS[name]
            self._loaded_vlms[name] = extractor_cls(
                max_new_tokens=self.max_new_tokens,
                **self.vlm_engine_kwargs.get(name, {}),
            )
        return self._loaded_vlms[name]

    # ------------------------------------------------------------------
    # Already-processed set
    # ------------------------------------------------------------------

    def _load_done_keys(self) -> set:
        """Return the set of (image_stem, config, detector, article_idx, vlm) tuples already done."""
        if not self.parquet_path.exists():
            return set()
        df = pd.read_parquet(self.parquet_path, columns=_KEY_COLS + ["status"])
        if self.skip_failed_crops:
            df = df[df["status"] == "ok"]
        return set(zip(*(df[c] for c in _KEY_COLS)))

    # ------------------------------------------------------------------
    # Crop discovery
    # ------------------------------------------------------------------

    def _discover_crops(self, crops_root: Path) -> List[Dict[str, Any]]:
        """
        Read the layout Parquet and return one metadata dict per
        successfully-laid-out crop that exists on disk. Image loading is
        deferred to batch time so we don't hold every crop in memory at
        once.
        """
        layout_df = pd.read_parquet(self.layout_parquet)
        layout_df = layout_df[layout_df["status"] == "ok"].copy()

        crops = []
        for _, layout_row in layout_df.iterrows():
            crop_path = (
                crops_root
                / layout_row["image_stem"]
                / f"config_{layout_row['preprocessing_config']}"
                / layout_row["detector"]
                / layout_row["crop_file"]
            )
            if not crop_path.exists():
                self.logger.warning("Crop file missing: %s", crop_path)
                continue

            crops.append(
                {
                    "crop_path": crop_path,
                    "newspaper": layout_row["newspaper"],
                    "date": layout_row["date"],
                    "page": layout_row["page"],
                    "image_stem": layout_row["image_stem"],
                    "preprocessing_config": int(layout_row["preprocessing_config"]),
                    "detector": layout_row["detector"],
                    "article_idx": int(layout_row["article_idx"]),
                    "crop_file": layout_row["crop_file"],
                    "x1": int(layout_row["x1"]),
                    "y1": int(layout_row["y1"]),
                    "x2": int(layout_row["x2"]),
                    "y2": int(layout_row["y2"]),
                    "grid_row": int(layout_row["grid_row"]),
                    "grid_col": int(layout_row["grid_col"]),
                }
            )
        return crops

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self, crops_root: Path) -> int:
        """
        Extract text from all article crops listed in the layout Parquet,
        one VLM at a time, batched.

        Parameters
        ----------
        crops_root : Root directory where layout analysis saved its crop TIFFs.
                     Crop files are looked up as:
                     crops_root / image_stem / config_{N} / detector / crop_file

        Returns
        -------
        int : Number of (crop, VLM) extractions successfully attempted.
        """
        if not self.layout_parquet.exists():
            self.logger.error(
                "Layout Parquet not found: %s — run LayoutAnalysisPipeline first.",
                self.layout_parquet,
            )
            return 0

        crops = self._discover_crops(crops_root)
        if not crops:
            self.logger.warning("No usable crops found under %s", crops_root)
            return 0

        done_keys = self._load_done_keys()
        self.logger.info(
            "Crops available: %d. Already-extracted (any vlm): %d.",
            len(crops), len(done_keys),
        )

        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        processed = 0

        for vlm_name in self.vlm_names:
            pending = [
                crop
                for crop in crops
                if (
                    crop["image_stem"],
                    crop["preprocessing_config"],
                    crop["detector"],
                    crop["article_idx"],
                    vlm_name,
                )
                not in done_keys
            ]
            if not pending:
                self.logger.info("[%s] nothing pending, skipping.", vlm_name)
                continue

            self.logger.info("[%s] %d crops pending, loading model...", vlm_name, len(pending))
            extractor = self._get_vlm(vlm_name)

            for chunk_start in range(0, len(pending), self.batch_size):
                chunk = pending[chunk_start : chunk_start + self.batch_size]
                images, metadata_list = [], []
                for crop in chunk:
                    image = cv2.imread(str(crop["crop_path"]))
                    if image is None:
                        self.logger.warning("Could not read crop: %s", crop["crop_path"])
                        continue
                    metadata = {k: v for k, v in crop.items() if k != "crop_path"}
                    metadata["vlm"] = vlm_name
                    images.append(image)
                    metadata_list.append(metadata)

                if not images:
                    continue

                self.logger.info(
                    "[%s] batch %d-%d / %d",
                    vlm_name,
                    chunk_start,
                    chunk_start + len(images),
                    len(pending),
                )

                results: List[ExtractionResult] = extractor.extract_batch(images, metadata_list)

                for result in results:
                    self.logger.info(
                        "  → status=%s | title=%r | body_chars=%d | %.3fs",
                        result.status,
                        result.title[:60] if result.title else "",
                        len(result.body),
                        result.elapsed_s,
                    )

                self._append_to_parquet([r.to_dict() for r in results])
                processed += len(results)

            # Free GPU memory before loading the next VLM — vLLM does not
            # release device memory just because the object is dereferenced.
            self.logger.info("[%s] done, unloading model to free GPU memory.", vlm_name)
            extractor.unload()
            del self._loaded_vlms[vlm_name]

        self.logger.info("VLM extraction complete. Extractions run: %d", processed)
        return processed

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def _append_to_parquet(self, rows: List[Dict]) -> None:
        new_df = pd.DataFrame(rows)
        if self.parquet_path.exists():
            existing = pd.read_parquet(self.parquet_path)
            combined = pd.concat([existing, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=_KEY_COLS, keep="last")
        else:
            combined = new_df

        combined = combined.sort_values(
            ["newspaper", "date", "page", "preprocessing_config", "detector", "article_idx", "vlm"]
        ).reset_index(drop=True)
        combined.to_parquet(self.parquet_path, index=False)

        self.logger.info(
            "Saved %d new rows to %s (total: %d)",
            len(new_df), self.parquet_path, len(combined),
        )