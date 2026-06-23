"""
VLM extraction pipeline.

Reads article crop TIFFs produced by the layout analysis step and runs one
or more VLMs over each crop to extract title and body text.

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

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import pandas as pd

from .vlm_steps import (
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
    Runs one or more VLM extractors over every article crop produced by the
    layout analysis step.

    Usage
    -----
    The pipeline reads a layout Parquet file (produced by LayoutAnalysisPipeline)
    to discover which crop TIFFs exist and what metadata to attach.  It then
    iterates over each crop × VLM combination, calls the VLM, and appends
    results to a separate Parquet file.

    Previously processed (crop, vlm) pairs are skipped automatically so the
    pipeline is safe to re-run after interruption.

    Parameters
    ----------
    logger                : Python logger instance.
    vlms                  : List of VLM names to run (subset of VLM_EXTRACTORS).
    layout_parquet        : Path to the layout analysis results Parquet file.
    parquet_path          : Output Parquet path for VLM results.
    olmocr_server_url     : vLLM server URL for OlmOCR.
    rolmocr_server_url    : vLLM server URL for RolmOCR.
    nanonets_server_url   : vLLM server URL for NanonetsOCR.
    olmocr_use_local      : If True, load OlmOCR locally (no server needed).
    nanonets_use_local    : If True, load NanonetsOCR locally.
    max_new_tokens        : Passed to every VLM.
    skip_failed_crops     : If True, skip crops that previously failed.
    """

    def __init__(
        self,
        logger: logging.Logger,
        vlms: List[str] = None,
        layout_parquet: str = "data/corpus_construction/layout_detection/results.parquet",
        parquet_path: str = DEFAULT_PARQUET,
        olmocr_server_url: str = "http://localhost:8001/v1",
        rolmocr_server_url: str = "http://localhost:8002/v1",
        nanonets_server_url: str = "http://localhost:8003/v1",
        olmocr_use_local: bool = False,
        nanonets_use_local: bool = False,
        max_new_tokens: int = 2048,
        skip_failed_crops: bool = True,
    ):
        self.logger = logger
        self.vlm_names = vlms or list(VLM_EXTRACTORS.keys())
        self.layout_parquet = Path(layout_parquet)
        self.parquet_path = Path(parquet_path)
        self.olmocr_server_url = olmocr_server_url
        self.rolmocr_server_url = rolmocr_server_url
        self.nanonets_server_url = nanonets_server_url
        self.olmocr_use_local = olmocr_use_local
        self.nanonets_use_local = nanonets_use_local
        self.max_new_tokens = max_new_tokens
        self.skip_failed_crops = skip_failed_crops

        self._loaded_vlms: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # VLM cache
    # ------------------------------------------------------------------

    def _get_vlm(self, name: str):
        if name not in self._loaded_vlms:
            self.logger.info("Loading VLM extractor: %s", name)
            if name == "olmocr":
                self._loaded_vlms[name] = OlmOCRExtractor(
                    server_url=self.olmocr_server_url,
                    max_new_tokens=self.max_new_tokens,
                    use_local=self.olmocr_use_local,
                )
            elif name == "rolmocr":
                self._loaded_vlms[name] = RolmOCRExtractor(
                    server_url=self.rolmocr_server_url,
                    max_new_tokens=self.max_new_tokens,
                )
            elif name == "nanonets":
                self._loaded_vlms[name] = NanonetsOCRExtractor(
                    server_url=self.nanonets_server_url,
                    max_new_tokens=self.max_new_tokens,
                    use_local=self.nanonets_use_local,
                )
            else:
                raise ValueError(f"Unknown VLM: {name!r}. Choose from {list(VLM_EXTRACTORS)}")
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
    # Core run
    # ------------------------------------------------------------------

    def run(self, crops_root: Path) -> int:
        """
        Extract text from all article crops listed in the layout Parquet.

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

        layout_df = pd.read_parquet(self.layout_parquet)
        # Only process rows where layout succeeded and a crop file exists
        layout_df = layout_df[layout_df["status"] == "ok"].copy()

        if layout_df.empty:
            self.logger.warning("No successful layout rows found in %s", self.layout_parquet)
            return 0

        done_keys = self._load_done_keys()
        self.logger.info(
            "Layout crops available: %d. Already extracted: %d.",
            len(layout_df), len(done_keys),
        )

        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        processed = 0
        rows_buffer: List[Dict] = []

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

            image = cv2.imread(str(crop_path))
            if image is None:
                self.logger.warning("Could not read crop: %s", crop_path)
                continue

            metadata = {
                "newspaper": layout_row["newspaper"],
                "date": layout_row["date"],
                "page": layout_row["page"],
                "image_stem": layout_row["image_stem"],
                "preprocessing_config": int(layout_row["preprocessing_config"]),
                "detector": layout_row["detector"],
                "article_idx": int(layout_row["article_idx"]),
                "crop_file": layout_row["crop_file"],
                # Geometry — forwarded for convenience
                "x1": int(layout_row["x1"]),
                "y1": int(layout_row["y1"]),
                "x2": int(layout_row["x2"]),
                "y2": int(layout_row["y2"]),
                "grid_row": int(layout_row["grid_row"]),
                "grid_col": int(layout_row["grid_col"]),
            }

            for vlm_name in self.vlm_names:
                key = (
                    metadata["image_stem"],
                    metadata["preprocessing_config"],
                    metadata["detector"],
                    metadata["article_idx"],
                    vlm_name,
                )
                if key in done_keys:
                    self.logger.debug("Skipping already-done: %s", key)
                    continue

                self.logger.info(
                    "[%s] config_%s | detector=%s | article=%d | vlm=%s",
                    metadata["image_stem"],
                    metadata["preprocessing_config"],
                    metadata["detector"],
                    metadata["article_idx"],
                    vlm_name,
                )

                extractor = self._get_vlm(vlm_name)
                result = extractor.extract(image, metadata=metadata)

                self.logger.info(
                    "  → status=%s | title=%r | body_chars=%d | %.3fs",
                    result.status,
                    result.title[:60] if result.title else "",
                    len(result.body),
                    result.elapsed_s,
                )

                rows_buffer.append(result.to_dict())
                processed += 1

                # Flush to Parquet every 50 results to limit data loss on crash
                if len(rows_buffer) >= 50:
                    self._append_to_parquet(rows_buffer)
                    rows_buffer.clear()

        if rows_buffer:
            self._append_to_parquet(rows_buffer)

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