"""Run OCR backends over successful binarization outputs."""

import atexit
import logging
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

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
    """
    Initialize exactly one OCR extractor per worker process.

    Each worker gets its own model instance and reuses it for
    all images assigned to that process.
    """
    global _WORKER_EXTRACTOR
    global _WORKER_EXTRACTOR_NAME

    # Prevent libraries such as MKL/OpenBLAS/PyTorch from creating
    # additional CPU threads inside each worker.
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
            # Can happen if inter-op threads were already initialized.
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

    # Ensure cleanup when the worker process exits.
    atexit.register(_close_worker_extractor)


def _process_one(item: dict) -> dict:
    """Run OCR for one image using the worker-local extractor."""
    if _WORKER_EXTRACTOR is None:
        raise RuntimeError("OCR worker extractor has not been initialized.")

    start = time.time()

    row = dict(
        item,
        ocr_extractor=_WORKER_EXTRACTOR_NAME,
    )

    try:
        text = _WORKER_EXTRACTOR.extract(item["image_path"])

        row.update(
            text=text,
            elapsed_s=time.time() - start,
            status="success",
            error=None,
        )

    except Exception as exc:
        row.update(
            text="",
            elapsed_s=time.time() - start,
            status="failed",
            error=str(exc),
        )

    return row


class OCRExtractionPipeline:
    def __init__(
        self,
        logger: logging.Logger,
        extractors,
        binarization_parquet,
        binarized_dir,
        parquet_path,
        skip_failed=True,
        workers=1,
    ):
        self.logger = logger
        self.extractor_names = extractors
        self.binarization_parquet = Path(binarization_parquet)
        self.binarized_dir = Path(binarized_dir)
        self.parquet_path = Path(parquet_path)
        self.skip_failed = skip_failed
        self.workers = max(1, int(workers))

    def _load_done(self):
        if not self.parquet_path.exists():
            return set()

        try:
            df = pd.read_parquet(self.parquet_path)
        except Exception as exc:
            self.logger.warning(
                "Could not read existing OCR Parquet: %s",
                exc,
            )
            return set()

        if not set(KEY_COLS + ["status"]).issubset(df.columns):
            return set()

        if self.skip_failed:
            df = df[df.status == "success"]

        return set(zip(*(df[c] for c in KEY_COLS)))

    def _discover_inputs(self):
        df = pd.read_parquet(self.binarization_parquet)

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
                "Binarization Parquet is missing required columns: "
                f"{sorted(missing)}"
            )

        df = df[df.status == "success"]

        inputs = []

        for _, row in df.iterrows():
            relative_path = str(row.binarize_file)
            image_path = self.binarized_dir / relative_path

            if not image_path.exists():
                self.logger.warning(
                    "Binarized file missing: %s",
                    image_path,
                )
                continue

            detector = (
                None
                if pd.isna(row.detector)
                else str(row.detector)
            )

            inputs.append(
                {
                    "image_path": str(image_path),
                    "image_stem": str(row.image_stem),
                    "config_id": int(row.config_id),
                    "layout_mode": (
                        "none"
                        if detector is None
                        else "layout"
                    ),
                    "detector": detector,
                    "binarization": str(row.binarization),
                    "binarize_file": relative_path,
                }
            )

        return inputs

    def _append_batch(self, rows: list):
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

        df = (
            pd.concat([old, new], ignore_index=True)
            .drop_duplicates(KEY_COLS, keep="last")
        )

        df["_detector_sort"] = (
            df.detector.fillna("").astype(str)
        )

        df = (
            df.sort_values(
                [
                    "image_stem",
                    "config_id",
                    "layout_mode",
                    "_detector_sort",
                    "binarization",
                    "ocr_extractor",
                ]
            )
            .drop(columns="_detector_sort")
        )

        self.parquet_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        df.to_parquet(
            self.parquet_path,
            index=False,
        )

    def _run_extractor(self, name, pending):
        """Run one OCR backend across all pending images."""
        self.logger.info(
            "Running %s OCR on %d images with %d worker processes.",
            name,
            len(pending),
            self.workers,
        )

        processed = 0
        pending_rows = []

        # spawn is safer for complex ML libraries than fork.
        mp_context = __import__(
            "multiprocessing"
        ).get_context("spawn")

        with ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp_context,
            initializer=_init_worker,
            initargs=(name,),
        ) as executor:

            # chunksize=1 gives good load balancing because OCR
            # runtimes can vary substantially between pages.
            results = executor.map(
                _process_one,
                pending,
                chunksize=1,
            )

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
                    pending_rows = []

            if pending_rows:
                self._append_batch(pending_rows)

        return processed

    def run(self):
        if not self.binarization_parquet.exists():
            raise FileNotFoundError(
                self.binarization_parquet
            )

        if self.workers < 1:
            raise ValueError(
                f"workers must be >= 1, got {self.workers}"
            )

        inputs = self._discover_inputs()
        done = self._load_done()

        processed = 0

        self.logger.info(
            "Discovered %d successful binarization outputs.",
            len(inputs),
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
                if (
                    tuple(
                        item[c]
                        for c in KEY_COLS[:-1]
                    )
                    + (name,)
                ) not in done
            ]

            if not pending:
                self.logger.info(
                    "Skipping %s: all jobs already completed.",
                    name,
                )
                continue

            processed += self._run_extractor(
                name,
                pending,
            )

        return processed