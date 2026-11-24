"""
pipeline.py

LLM extraction pipeline with newspaper prompt loading, LLM routing,
and year-sharded Parquet I/O.
"""

import logging
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .steps import ExtractionResult, LLM_EXTRACTORS

DEFAULT_PROMPTS_DIR = Path("data/corpus_construction/local_dspy_llm/prompts")
DEFAULT_PARQUET = "data/corpus_construction/llm_extraction/results.parquet"

# Default winning LLM(s) per newspaper (override anytime via --llms or --all-llms)
NEWSPAPER_LLMS: Dict[str, List[str]] = {
    "correo": ["qwen"],
    "elcomercio": ["qwen"],
    "gestion": ["qwen"],
    "ojo": ["qwen"],
    "peru21": ["qwen"],
    "publimetro": ["qwen"],
    "trome": ["qwen"],
}

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


def _compute_batch_size(
    max_batch_size: int = 64,
    min_batch_size: int = 1,
    tokens_per_item: int = 4096,
    bytes_per_token: int = 2,
) -> int:
    """Estimate a safe batch size from available GPU memory."""
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
        free_bytes = free_mib * 1024 * 1024

        usable_bytes = free_bytes * 0.8
        bytes_per_item = tokens_per_item * bytes_per_token
        batch_size = max(min_batch_size, int(usable_bytes / bytes_per_item))
        return min(batch_size, max_batch_size)
    except Exception:
        return min_batch_size


def _wait_for_gpu_memory(
    logger: logging.Logger,
    min_free_gib: float = 36.0,
    poll_interval_s: float = 5.0,
    timeout_s: float = 30.0,
) -> None:
    """Wait until enough GPU memory is free before loading the next LLM."""
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


class LLMExtractionPipeline:
    """
    Run local in-process LLM structured extraction over successful OCR
    outputs.
    """

    def __init__(
        self,
        logger: logging.Logger,
        llms: Optional[List[str]] = None,
        ocr_parquet: Optional[str] = None,
        ocr_dir: Optional[str] = None,
        parquet_path: Optional[str] = None,
        output_dir: Optional[str] = None,
        sub_dirs: Optional[List[str]] = None,
        newspaper: Optional[str] = None,
        prompts_dir: Optional[str] = None,
        max_new_tokens: int = 4096,
        min_free_gib: float = 36.0,
        skip_failed_extractions: bool = True,
        llm_engine_kwargs: Optional[Dict[str, Dict[str, Any]]] = None,
    ):
        self.logger = logger
        self.llm_names = llms or list(LLM_EXTRACTORS.keys())
        self.ocr_parquet = Path(ocr_parquet) if ocr_parquet else None
        self.ocr_dir = Path(ocr_dir) if ocr_dir else None
        self.sub_dirs = list(sub_dirs) if sub_dirs else None
        self.newspaper = newspaper.lower() if newspaper else None
        self.prompts_dir = (
            Path(prompts_dir) if prompts_dir else DEFAULT_PROMPTS_DIR
        )
        self.max_new_tokens = max_new_tokens
        self.min_free_gib = min_free_gib
        self.skip_failed_extractions = skip_failed_extractions
        self.llm_engine_kwargs = llm_engine_kwargs or {}
        self._loaded_llms: Dict[str, Any] = {}

        # Resolve newspaper-specific markdown prompt if present
        self.prompt_path: Optional[Path] = None
        if self.newspaper:
            candidate = self.prompts_dir / f"{self.newspaper}.md"
            if candidate.exists():
                self.prompt_path = candidate
                self.logger.info(
                    "Using newspaper prompt for %s: %s",
                    self.newspaper,
                    self.prompt_path,
                )
            else:
                self.logger.warning(
                    "No custom prompt found at %s; falling back to default template.",
                    candidate,
                )

        # Resolve shard-safe output Parquet path
        if parquet_path:
            self.parquet_path = Path(parquet_path)
        elif output_dir:
            llm_tag = "_".join(sorted(self.llm_names))
            sub_tag = f"_{'_'.join(self.sub_dirs)}" if self.sub_dirs else ""
            self.parquet_path = (
                Path(output_dir) / f"llm{sub_tag}_{llm_tag}.parquet"
            )
        else:
            self.parquet_path = Path(DEFAULT_PARQUET)

    # ------------------------------------------------------------------
    # LLM cache
    # ------------------------------------------------------------------

    def _get_llm(self, name: str):
        """Load an LLM lazily and cache it with the newspaper's prompt."""
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
                prompt_path=self.prompt_path,
                **self.llm_engine_kwargs.get(name, {}),
            )

        return self._loaded_llms[name]

    # ------------------------------------------------------------------
    # Resume: already-processed keys
    # ------------------------------------------------------------------

    def _load_done_keys(self) -> set:
        """Return the set of already-completed extraction keys."""
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

    def _resolve_ocr_parquets(self) -> List[Path]:
        """Resolve single or year-sharded OCR Parquet file(s)."""
        if self.ocr_parquet is not None:
            if not self.ocr_parquet.exists():
                raise FileNotFoundError(self.ocr_parquet)
            return [self.ocr_parquet]

        if self.ocr_dir is None or not self.ocr_dir.exists():
            raise FileNotFoundError(
                f"OCR directory does not exist: {self.ocr_dir}"
            )

        all_parquets = sorted(self.ocr_dir.glob("ocr*.parquet"))
        if not all_parquets:
            raise FileNotFoundError(
                f"No ocr*.parquet files found in {self.ocr_dir}"
            )

        if not self.sub_dirs:
            return all_parquets

        matched = [
            p
            for p in all_parquets
            if any(
                f"_{sd}_" in p.name or p.name.endswith(f"_{sd}.parquet")
                for sd in self.sub_dirs
            )
        ]
        return matched if matched else all_parquets

    def _discover_inputs(self) -> List[Dict[str, Any]]:
        """Read single or sharded OCR Parquet(s) and return successful rows."""
        parquet_files = self._resolve_ocr_parquets()
        self.logger.info(
            "Loading OCR metadata from %d Parquet file(s)",
            len(parquet_files),
        )

        df = pd.concat(
            [pd.read_parquet(p) for p in parquet_files],
            ignore_index=True,
        )

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

        df = df[df["status"] == "success"].drop_duplicates(
            subset=_KEY_COLS[:-1],
            keep="last",
        )

        if self.sub_dirs:
            year_tokens = tuple(f"_{sd}-" for sd in self.sub_dirs)
            df = df[
                df["image_stem"].astype(str).map(
                    lambda s: any(tok in s for tok in year_tokens)
                )
            ]

        inputs = []
        for row in df.itertuples(index=False):
            detector = None if pd.isna(row.detector) else str(row.detector)
            inputs.append(
                {
                    "image_stem": str(row.image_stem),
                    "config_id": int(row.config_id),
                    "layout_mode": str(row.layout_mode),
                    "detector": detector,
                    "binarization": str(row.binarization),
                    "binarize_file": str(row.binarize_file),
                    "ocr_extractor": str(row.ocr_extractor),
                    "ocr_text": str(row.text),
                }
            )

        return inputs

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self) -> int:
        """Run all requested LLMs over all successful OCR outputs."""
        inputs = self._discover_inputs()

        if not inputs:
            self.logger.warning("No usable OCR rows found.")
            return 0

        done_keys = self._load_done_keys()

        self.logger.info(
            "OCR rows available: %d (output=%s)",
            len(inputs),
            self.parquet_path,
        )
        self.logger.info(
            "Already completed extractions: %d",
            len(done_keys),
        )

        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)

        processed = 0

        for llm_name in self.llm_names:
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

            batch_size = _compute_batch_size(
                max_batch_size=64,
                tokens_per_item=self.max_new_tokens,
                bytes_per_token=1,
            )
            self.logger.info("[%s] Dynamic batch size: %d", llm_name, batch_size)

            for chunk_start in range(0, len(pending), batch_size):
                chunk = pending[chunk_start : chunk_start + batch_size]
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

                results: List[ExtractionResult] = extractor.extract_batch(
                    ocr_texts, metadata_list
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

                processed += sum(max(1, len(r.articles)) for r in results)

                for result in results:
                    if result.status != "success":
                        continue
                    key = tuple(result.metadata.get(c) for c in _KEY_COLS)
                    done_keys.add(key)

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
        """Append extraction results to the output Parquet."""
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