"""Run OCR backends over successful binarization outputs with newspaper routing and sharding."""

import atexit
import logging
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd

from .registry import OCR_EXTRACTORS


KEY_COLS = [
    "image_stem",
    "config_id",
    "layout_mode",
    "detector",
    "binarization",
    "binarize_file",
    "ocr_extractor",
]

FLUSH_EVERY = 50

# Winning OCR extractor(s) per newspaper (verify against evaluation config)
NEWSPAPER_OCR_EXTRACTORS: Dict[str, List[str]] = {
    "correo": ["docling_rapidocr"],
    "elcomercio": ["docling_rapidocr"],
    "gestion": ["docling_rapidocr"],
    "ojo": ["docling_rapidocr"],
    "peru21": ["docling_rapidocr"],
    "publimetro": ["docling_easyocr"],
    "trome": ["docling_rapidocr", "docling_rapidocr"],
}

# Optional position-specific config_id -> OCR extractor routing for Trome
TROME_CONFIG_OCR: Dict[int, str] = {
    7: "docling_rapidocr",     # Last page
    34: "docling_rapidocr",   # Interior / other pages
}

# Worker-local state. Each process gets exactly one extractor/model.
_WORKER_EXTRACTOR = None
_WORKER_EXTRACTOR_NAME = None


def _close_worker_extractor():
    """Close the extractor owned by this worker process."""
    global _WORKER_EXTRACTOR

    if _WORKER_EXTRACTOR is not None:
        try:
            _WORKER_EXTRACTOR.close()
        except Exception:
            pass

    _WORKER_EXTRACTOR = None


def _init_worker(extractor_name: str):
    """Initialize one OCR extractor per worker process and cap CPU thread oversubscription."""
    global _WORKER_EXTRACTOR
    global _WORKER_EXTRACTOR_NAME

    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    os.environ["NUMEXPR_NUM_THREADS"] = "1"

    try:
        import torch

        torch.set_num_threads(1)
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass
    except Exception:
        pass

    if extractor_name not in OCR_EXTRACTORS:
        raise ValueError(
            f"Unknown OCR extractor {extractor_name!r}; "
            f"choose from {list(OCR_EXTRACTORS)}"
        )

    _WORKER_EXTRACTOR_NAME = extractor_name
    _WORKER_EXTRACTOR = OCR_EXTRACTORS[extractor_name]()
    atexit.register(_close_worker_extractor)


def _process_one(item: dict) -> dict:
    """Run OCR for one image using the worker-local extractor."""
    if _WORKER_EXTRACTOR is None:
        raise RuntimeError("OCR worker extractor has not been initialized.")

    start = time.time()
    row = dict(item, ocr_extractor=_WORKER_EXTRACTOR_NAME)

    try:
        text = _WORKER_EXTRACTOR.extract(item["image_path"])
        row.update(
            text=text,
            elapsed_s=round(time.time() - start, 3),
            status="success",
            error=None,
        )
    except Exception as exc:
        row.update(
            text="",
            elapsed_s=round(time.time() - start, 3),
            status="failed",
            error=str(exc),
        )

    return row


class OCRExtractionPipeline:
    def __init__(
        self,
        logger: logging.Logger,
        extractors: List[str],
        binarized_dir: Path,
        output_dir: Path,
        binarization_parquet: Optional[Path] = None,
        sub_dirs: Optional[List[str]] = None,
        newspaper: Optional[str] = None,
        skip_failed: bool = True,
        workers: int = 1,
        batch_size: int = 8,
    ):
        self.logger = logger
        self.extractor_names = list(extractors)
        self.binarized_dir = Path(binarized_dir)
        self.output_dir = Path(output_dir)
        self.binarization_parquet = (
            Path(binarization_parquet) if binarization_parquet else None
        )
        self.sub_dirs = list(sub_dirs) if sub_dirs else None
        self.newspaper = newspaper.lower() if newspaper else None
        self.skip_failed = skip_failed
        self.workers = max(1, int(workers))
        self.batch_size = max(1, int(batch_size))

        ext_tag = "_".join(sorted(self.extractor_names))
        sub_tag = f"_{'_'.join(self.sub_dirs)}" if self.sub_dirs else ""
        self.parquet_path = (
            self.output_dir / f"ocr{sub_tag}_{ext_tag}.parquet"
        )

    def _extractors_for_item(self, config_id: int) -> List[str]:
        """Resolve which active OCR extractor(s) apply to a given (newspaper, config_id)."""
        if (
            self.newspaper == "trome"
            and config_id in TROME_CONFIG_OCR
        ):
            target = TROME_CONFIG_OCR[config_id]
            return [e for e in self.extractor_names if e == target]
        return self.extractor_names

    def _resolve_binarization_parquets(self) -> List[Path]:
        """Find all relevant binarization Parquet file(s) for the requested shard."""
        if self.binarization_parquet is not None:
            if not self.binarization_parquet.exists():
                raise FileNotFoundError(self.binarization_parquet)
            return [self.binarization_parquet]

        all_parquets = sorted(self.binarized_dir.glob("binarization*.parquet"))
        if not all_parquets:
            raise FileNotFoundError(
                f"No binarization*.parquet files found in {self.binarized_dir}"
            )

        if not self.sub_dirs:
            return all_parquets

        # Match shard-specific parquets like binarization_2014_otsu.parquet
        matched = [
            p
            for p in all_parquets
            if any(f"_{sd}_" in p.name or p.name.endswith(f"_{sd}.parquet") for sd in self.sub_dirs)
        ]
        return matched if matched else all_parquets

    def _load_done(self) -> set:
        if not self.parquet_path.exists():
            return set()

        try:
            df = pd.read_parquet(self.parquet_path)
        except Exception as exc:
            self.logger.warning("Could not read existing OCR Parquet: %s", exc)
            return set()

        if not set(KEY_COLS + ["status"]).issubset(df.columns):
            return set()

        if self.skip_failed:
            df = df[df.status == "success"]

        return set(zip(*(df[c] for c in KEY_COLS)))

    def _discover_inputs(self) -> List[dict]:
        parquet_files = self._resolve_binarization_parquets()
        self.logger.info(
            "Loading binarization metadata from %d Parquet file(s) in %s",
            len(parquet_files),
            self.binarized_dir,
        )

        dfs = [pd.read_parquet(p) for p in parquet_files]
        df = pd.concat(dfs, ignore_index=True)

        required = {
            "image_stem",
            "config_id",
            "detector",
            "binarization",
            "binarize_file",
            "status",
        }
        missing = required - set(df.columns)
        if missing:
            raise ValueError(
                f"Binarization Parquet is missing required columns: {sorted(missing)}"
            )

        df = df[df.status == "success"].drop_duplicates(
            subset=["image_stem", "detector", "config_id", "binarization", "binarize_file"],
            keep="last",
        )

        # If sub_dirs were passed and a monolithic parquet was loaded, filter by year prefix
        if self.sub_dirs:
            year_tokens = tuple(f"_{sd}-" for sd in self.sub_dirs)
            df = df[
                df["image_stem"].astype(str).map(
                    lambda s: any(tok in s for tok in year_tokens)
                )
            ]

        inputs: List[dict] = []
        for row in df.itertuples(index=False):
            relative_path = str(row.binarize_file)
            image_path = self.binarized_dir / relative_path

            if not image_path.exists():
                self.logger.warning("Binarized file missing: %s", image_path)
                continue

            detector = None if pd.isna(row.detector) else str(row.detector)
            inputs.append(
                {
                    "image_path": str(image_path),
                    "image_stem": str(row.image_stem),
                    "config_id": int(row.config_id),
                    "layout_mode": "none" if detector is None else "cropped",
                    "detector": detector,
                    "binarization": str(row.binarization),
                    "binarize_file": relative_path,
                }
            )

        return inputs

    def _append_batch(self, rows: List[dict]) -> None:
        """Write a batch using one read/merge/write cycle."""
        if not rows:
            return

        new = pd.DataFrame(rows)

        if self.parquet_path.exists():
            try:
                old = pd.read_parquet(self.parquet_path)
                if not set(KEY_COLS).issubset(old.columns):
                    old = pd.DataFrame()
            except Exception:
                old = pd.DataFrame()
        else:
            old = pd.DataFrame()

        df = pd.concat([old, new], ignore_index=True).drop_duplicates(
            KEY_COLS, keep="last"
        )

        df["_detector_sort"] = df.detector.fillna("").astype(str)
        df = df.sort_values(
            [
                "image_stem",
                "config_id",
                "layout_mode",
                "_detector_sort",
                "binarization",
                "ocr_extractor",
            ]
        ).drop(columns="_detector_sort")

        self.parquet_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(self.parquet_path, index=False)

    def _run_extractor_in_process(self, name: str, pending: List[dict]) -> int:
        """Run one OCR backend in-process using Docling's batch converter (optimal for GPU)."""
        self.logger.info(
            "Running %s OCR in-process (batch_size=%d) on %d images.",
            name,
            self.batch_size,
            len(pending),
        )
        _init_worker(name)
        processed = 0
        pending_rows: List[dict] = []

        try:
            for start_idx in range(0, len(pending), self.batch_size):
                chunk = pending[start_idx : start_idx + self.batch_size]
                paths = [item["image_path"] for item in chunk]
                batch_start = time.time()

                try:
                    texts = _WORKER_EXTRACTOR.extract_batch(paths)
                    per_item_s = round(
                        (time.time() - batch_start) / max(1, len(chunk)), 3
                    )
                    for item, text in zip(chunk, texts):
                        row = dict(
                            item,
                            ocr_extractor=name,
                            text=text,
                            elapsed_s=per_item_s,
                            status="success",
                            error=None,
                        )
                        pending_rows.append(row)
                        processed += 1
                except Exception:
                    # Fallback to per-image extraction so one corrupt crop doesn't fail the batch
                    for item in chunk:
                        row = _process_one(item)
                        pending_rows.append(row)
                        processed += 1

                if len(pending_rows) >= FLUSH_EVERY:
                    self._append_batch(pending_rows)
                    pending_rows.clear()
                    self.logger.info(
                        "[%s] Progress: %d/%d images completed",
                        name,
                        processed,
                        len(pending),
                    )

            if pending_rows:
                self._append_batch(pending_rows)
        finally:
            _close_worker_extractor()

        return processed

    def _run_extractor_multiprocess(self, name: str, pending: List[dict]) -> int:
        """Run one OCR backend across multiple spawned worker processes."""
        self.logger.info(
            "Running %s OCR on %d images with %d worker processes.",
            name,
            len(pending),
            self.workers,
        )

        processed = 0
        pending_rows: List[dict] = []
        mp_context = multiprocessing.get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(name,),
        ) as executor:
            results = executor.map(_process_one, pending, chunksize=1)

            for i, row in enumerate(results, start=1):
                processed += 1
                pending_rows.append(row)

                if row["status"] == "success":
                    self.logger.info(
                        "[%s] %d/%d done in %.1fs: %s",
                        name,
                        i,
                        len(pending),
                        row["elapsed_s"],
                        row["image_stem"],
                    )
                else:
                    self.logger.error(
                        "[%s] OCR failed for %s: %s",
                        name,
                        row["image_path"],
                        row["error"],
                    )

                if len(pending_rows) >= FLUSH_EVERY:
                    self._append_batch(pending_rows)
                    pending_rows.clear()

            if pending_rows:
                self._append_batch(pending_rows)

        return processed

    def run(self) -> int:
        inputs = self._discover_inputs()
        done = self._load_done()
        processed = 0

        self.logger.info(
            "Discovered %d successful binarization outputs (output=%s).",
            len(inputs),
            self.parquet_path,
        )

        for name in self.extractor_names:
            if name not in OCR_EXTRACTORS:
                raise ValueError(
                    f"Unknown OCR extractor {name!r}; "
                    f"choose from {list(OCR_EXTRACTORS)}"
                )

            pending = [
                item
                for item in inputs
                if name in self._extractors_for_item(item["config_id"])
                and (tuple(item[c] for c in KEY_COLS[:-1]) + (name,)) not in done
            ]

            if not pending:
                self.logger.info(
                    "Skipping %s: all jobs already completed.",
                    name,
                )
                continue

            if self.workers == 1:
                processed += self._run_extractor_in_process(name, pending)
            else:
                processed += self._run_extractor_multiprocess(name, pending)

        return processed