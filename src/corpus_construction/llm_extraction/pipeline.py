"""
pipeline.py

LLM extraction pipeline.

Reads successful OCR rows from the OCR extraction Parquet and runs one
or more local text-only LLMs over each OCR text to produce structured
article fields.

Architecture:

    OCR parquet (text / markdown column)
        ↓
    local in-process LLM via vLLM + DSPy Predict
        ↓
    title / subheadline / author / body
        ↓
    global Parquet

No vLLM server is required.

Each LLM receives every successful OCR row as a separate input. This
allows evaluation across:

    config_id
        × layout_mode
        × detector
        × binarization
        × ocr_extractor
        × llm_extractor

Output schema
-------------

image_stem
config_id
layout_mode
detector
binarization
binarize_file
ocr_extractor
llm_extractor
article_index
title
subheadline
author
body
raw_text
elapsed_s
status
error
"""

import gc
import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .steps import ExtractionResult, LLM_EXTRACTORS

def _compute_batch_size(
    max_batch_size: int = 64,
    min_batch_size: int = 1,
    tokens_per_item: int = 4096,
    bytes_per_token: int = 2,  # bfloat16
) -> int:
    """Estimate a safe batch size from available GPU memory.
    
    Falls back to min_batch_size if nvidia-smi is unavailable.
    """
    try:
        import subprocess
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.free",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, check=True,
        )
        free_mib = int(result.stdout.strip().splitlines()[0])
        free_bytes = free_mib * 1024 * 1024

        # Reserve 20% as safety margin
        usable_bytes = free_bytes * 0.8
        bytes_per_item = tokens_per_item * bytes_per_token
        batch_size = max(min_batch_size, int(usable_bytes / bytes_per_item))
        return min(batch_size, max_batch_size)

    except Exception:
        return min_batch_size

def _wait_for_gpu_memory(
    logger: logging.Logger,
    min_free_gib: float = 60.0,
    poll_interval_s: float = 5.0,
    timeout_s: float = 20.0,
) -> None:
    """
    Wait until enough GPU memory is free before loading the next LLM.

    This is intentionally based on nvidia-smi rather than PyTorch's
    memory accounting, because the latter only describes allocations
    visible to the current process.
    """
    min_free_mib = int(min_free_gib * 1024)
    deadline = time.monotonic() + timeout_s

    while True:
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.free",
                    "--format=csv,noheader,nounits",
                ],
                capture_output=True,
                text=True,
                check=True,
            )

            free_mib = int(result.stdout.strip().splitlines()[0])
        except (subprocess.SubprocessError, ValueError, IndexError) as exc:
            logger.warning(
                "Could not query GPU memory with nvidia-smi: %s",
                exc,
            )

            if time.monotonic() >= deadline:
                raise RuntimeError(
                    "Timed out while waiting for GPU memory."
                ) from exc

            time.sleep(poll_interval_s)
            continue

        logger.info(
            "GPU free memory: %.2f GiB / %.2f GiB required",
            free_mib / 1024,
            min_free_mib / 1024,
        )

        if free_mib >= min_free_mib:
            logger.info("GPU has enough free memory.")
            return
            
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"Timed out waiting for GPU memory. "
                f"Only {free_mib / 1024:.2f} GiB is free; "
                f"{min_free_mib / 1024:.2f} GiB required."
            )
        
        time.sleep(poll_interval_s)

        # else:
        #     logger.info(
        #         "Waiting for GPU memory to free up. "
        #         "Currently %.2f GiB free; %.2f GiB required.",
        #         free_mib / 1024,
        #         min_free_mib / 1024,
        #     )
        #     

        

DEFAULT_PARQUET = (
    "data/corpus_construction/llm_extraction/results.parquet"
)

# Unique extraction key — mirrors OCR key cols plus the LLM name.
_KEY_COLS = [
    "image_stem",
    "config_id",
    "layout_mode",
    "detector",
    "binarization",
    "binarize_file",
    "ocr_extractor",
    "llm_extractor",
]


class LLMExtractionPipeline:
    """
    Run local in-process LLM structured extraction over successful OCR
    outputs.

    The OCR Parquet is the metadata contract between the OCR and LLM
    stages. Each successful row represents one unique OCR text to
    process.
    """

    def __init__(
        self,
        logger: logging.Logger,
        llms: Optional[List[str]] = None,
        ocr_parquet: str = "",
        parquet_path: str = DEFAULT_PARQUET,
        max_new_tokens: int = 4096,
        min_free_gib: float = 60.0,
        skip_failed_extractions: bool = True,
        llm_engine_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.logger = logger
        self.llm_names = llms or list(LLM_EXTRACTORS.keys())
        self.ocr_parquet = Path(ocr_parquet)
        self.parquet_path = Path(parquet_path)
        self.max_new_tokens = max_new_tokens
        self.min_free_gib = min_free_gib
        self.skip_failed_extractions = skip_failed_extractions
        self.llm_engine_kwargs = llm_engine_kwargs or {}
        self._loaded_llms: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # LLM cache
    # ------------------------------------------------------------------

    def _get_llm(self, name: str):
        """Load a LLM lazily and cache it."""
        if name not in self._loaded_llms:
            if name not in LLM_EXTRACTORS:
                raise ValueError(
                    f"Unknown LLM: {name!r}. "
                    f"Choose from {list(LLM_EXTRACTORS)}"
                )

            self.logger.info(
                "Loading local LLM extractor: %s",
                name,
            )

            cls = LLM_EXTRACTORS[name]

            self._loaded_llms[name] = cls(
                max_new_tokens=self.max_new_tokens,
                **self.llm_engine_kwargs.get(name, {}),
            )

        return self._loaded_llms[name]

    # ------------------------------------------------------------------
    # Resume: already-processed keys
    # ------------------------------------------------------------------

    def _load_done_keys(self) -> set:
        """
        Return the set of already-completed extraction keys.

        Failed rows are excluded when skip_failed_extractions=True so
        they are retried on the next run.
        """
        if not self.parquet_path.exists():
            return set()

        try:
            available = set(
                pd.read_parquet(
                    self.parquet_path,
                    engine="pyarrow",
                ).columns
            )
        except Exception as exc:
            self.logger.warning(
                "Could not inspect existing LLM Parquet %s: %s",
                self.parquet_path,
                exc,
            )
            return set()

        required = set(_KEY_COLS + ["status"])
        missing = required - available

        if missing:
            self.logger.warning(
                "Existing LLM Parquet is incompatible with the current "
                "schema. Missing columns: %s. "
                "Existing results will not be used for resume.",
                sorted(missing),
            )
            return set()

        df = pd.read_parquet(
            self.parquet_path,
            columns=_KEY_COLS + ["status"],
        )

        if self.skip_failed_extractions:
            df = df[df["status"] == "success"]

        return set(zip(*(df[c] for c in _KEY_COLS)))

    # ------------------------------------------------------------------
    # Input discovery
    # ------------------------------------------------------------------

    def _discover_inputs(self) -> List[Dict[str, Any]]:
        """
        Read the OCR Parquet and return all successful rows.

        Each row becomes one LLM extraction input. The OCR text is
        carried in the 'text' column (markdown format).
        """
        df = pd.read_parquet(self.ocr_parquet)

        required = {
            "image_stem",
            "config_id",
            "layout_mode",
            "detector",
            "binarization",
            "binarize_file",
            "ocr_extractor",
            "text",
            "status",
        }

        missing = required - set(df.columns)

        if missing:
            raise ValueError(
                "OCR Parquet is missing required columns: "
                f"{sorted(missing)}"
            )

        df = df[df["status"] == "success"].copy()

        inputs = []

        for _, row in df.iterrows():
            detector = row["detector"]

            if pd.isna(detector):
                detector = None

            inputs.append(
                {
                    "image_stem": str(row["image_stem"]),
                    "config_id": int(row["config_id"]),
                    "layout_mode": str(row["layout_mode"]),
                    "detector": detector,
                    "binarization": str(row["binarization"]),
                    "binarize_file": str(row["binarize_file"]),
                    "ocr_extractor": str(row["ocr_extractor"]),
                    "ocr_text": str(row["text"]),
                }
            )

        return inputs

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self) -> int:
        """
        Run all requested LLMs over all successful OCR outputs.

        Returns
        -------
        int
            Number of article rows written.
        """
        if not self.ocr_parquet.exists():
            self.logger.error(
                "OCR Parquet not found: %s",
                self.ocr_parquet,
            )
            return 0

        inputs = self._discover_inputs()

        if not inputs:
            self.logger.warning("No usable OCR rows found.")
            return 0

        done_keys = self._load_done_keys()

        self.logger.info(
            "OCR rows available: %d",
            len(inputs),
        )
        self.logger.info(
            "Already completed extractions: %d",
            len(done_keys),
        )

        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)

        processed = 0

        for llm_name in self.llm_names:

            # ----------------------------------------------------------
            # Collect pending items for this LLM.
            # ----------------------------------------------------------

            pending = []

            for item in inputs:
                key = (
                    item["image_stem"],
                    item["config_id"],
                    item["layout_mode"],
                    item["detector"],
                    item["binarization"],
                    item["binarize_file"],
                    item["ocr_extractor"],
                    llm_name,
                )

                if key not in done_keys:
                    pending.append(item)

            if not pending:
                self.logger.info(
                    "[%s] Nothing pending.",
                    llm_name,
                )
                continue

            self.logger.info(
                "[%s] %d OCR texts pending.",
                llm_name,
                len(pending),
            )

            extractor = self._get_llm(llm_name)

            # ----------------------------------------------------------
            # Process in batches.
            # ----------------------------------------------------------

            batch_size = _compute_batch_size(
                max_batch_size=64,
                tokens_per_item=self.max_new_tokens,
                bytes_per_token=1,  # FP8 models use 1 byte/token for KV cache
            )
            self.logger.info("[%s] Dynamic batch size: %d", llm_name, batch_size)

            for chunk_start in range(
                0, len(pending), batch_size
            ):
                chunk = pending[
                    chunk_start: chunk_start + batch_size
                ]

                ocr_texts = [item["ocr_text"] for item in chunk]

                metadata_list = [
                    {
                        "image_stem": item["image_stem"],
                        "config_id": item["config_id"],
                        "layout_mode": item["layout_mode"],
                        "detector": item["detector"],
                        "binarization": item["binarization"],
                        "binarize_file": item["binarize_file"],
                        "ocr_extractor": item["ocr_extractor"],
                        "llm_extractor": llm_name,
                    }
                    for item in chunk
                ]

                self.logger.info(
                    "[%s] Batch %d-%d / %d",
                    llm_name,
                    chunk_start,
                    chunk_start + len(chunk),
                    len(pending),
                )

                results: List[ExtractionResult] = (
                    extractor.extract_batch(ocr_texts, metadata_list)
                )

                for result in results:
                    self.logger.info(
                        "status=%s | articles=%d | %.3fs",
                        result.status,
                        len(result.articles),
                        result.elapsed_s,
                    )
                    for article in result.articles:
                        self.logger.info(
                            "    [%d] %s",
                            article.article_index,
                            article.title[:80],
                        )

                rows = self._results_to_rows(results)

                self._append_to_parquet(rows)

                processed += sum(
                    max(1, len(r.articles)) for r in results
                )

                # Update in-memory resume set to avoid duplicate work
                # within the same run.
                for result in results:
                    if result.status != "success":
                        continue
                    key = tuple(
                        result.metadata.get(c)
                        for c in _KEY_COLS
                    )
                    done_keys.add(key)

            # ----------------------------------------------------------
            # Free GPU memory before loading the next LLM.
            # ----------------------------------------------------------

            self.logger.info(
                "[%s] Done. Unloading model.",
                llm_name,
            )
            
            extractor.unload()
            del self._loaded_llms[llm_name]
            
            _wait_for_gpu_memory(
                logger=self.logger,
                min_free_gib=self.min_free_gib,
            )

            self.logger.info(
                "[%s] Model unloaded.",
                llm_name,
            )

        self.logger.info(
            "LLM extraction complete. Extractions run: %d",
            processed,
        )

        return processed

    # ------------------------------------------------------------------
    # Row serialisation
    # ------------------------------------------------------------------

    @staticmethod
    def _results_to_rows(
        results: List[ExtractionResult],
    ) -> List[Dict]:
        """Flatten ExtractionResult objects into Parquet-ready dicts."""
        rows = []

        for result in results:
            base = dict(result.metadata)

            if result.status != "success":
                base.update(
                    {
                        "article_index": -1,
                        "title": "",
                        "subheadline": "",
                        "author": "",
                        "body": "",
                        "raw_text": result.raw_text,
                        "elapsed_s": result.elapsed_s,
                        "status": result.status,
                        "error": result.error,
                    }
                )
                rows.append(base)
                continue

            for article in result.articles:
                row = dict(base)
                row.update(
                    {
                        "article_index": article.article_index,
                        "title": article.title,
                        "subheadline": article.subheadline,
                        "author": article.author,
                        "body": article.body,
                        "raw_text": result.raw_text,
                        "elapsed_s": result.elapsed_s,
                        "status": result.status,
                        "error": result.error,
                    }
                )
                rows.append(row)

        return rows

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def _append_to_parquet(self, rows: List[Dict]) -> None:
        """
        Append extraction results to the output Parquet.

        Existing rows with the same unique key are replaced by the
        newest result (deduplication via drop_duplicates keep='last').
        """
        if not rows:
            return

        new_df = pd.DataFrame(rows)

        if self.parquet_path.exists():
            try:
                existing = pd.read_parquet(self.parquet_path)

                if not set(_KEY_COLS).issubset(existing.columns):
                    self.logger.warning(
                        "Existing LLM Parquet has an incompatible "
                        "schema. Replacing it."
                    )
                    existing = pd.DataFrame()

            except Exception as exc:
                self.logger.warning(
                    "Could not read existing LLM Parquet: %s. "
                    "Replacing it.",
                    exc,
                )
                existing = pd.DataFrame()

            combined = pd.concat(
                [existing, new_df],
                ignore_index=True,
            )

            combined = combined.drop_duplicates(
                subset=_KEY_COLS + ["article_index"],
                keep="last",
            )

        else:
            combined = new_df

        # Stable sort order for reproducibility.
        combined["_detector_sort"] = (
            combined["detector"].fillna("").astype(str)
        )

        combined = (
            combined.sort_values(
                [
                    "image_stem",
                    "config_id",
                    "layout_mode",
                    "_detector_sort",
                    "binarization",
                    "ocr_extractor",
                    "llm_extractor",
                ]
            )
            .drop(columns=["_detector_sort"])
            .reset_index(drop=True)
        )

        combined.to_parquet(self.parquet_path, index=False)

        self.logger.info(
            "Saved %d new rows to %s (total: %d)",
            len(new_df),
            self.parquet_path,
            len(combined),
        )