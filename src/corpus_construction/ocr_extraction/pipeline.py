"""Run OCR backends over successful binarization outputs."""

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
        self.workers = workers

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
                f"Binarization Parquet is missing required columns: "
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
                        "none" if detector is None else "layout"
                    ),
                    "detector": detector,
                    "binarization": str(row.binarization),
                    "binarize_file": relative_path,
                }
            )

        return inputs

    def _append(self, row):
        new = pd.DataFrame([row])

        if self.parquet_path.exists():
            try:
                old = pd.read_parquet(self.parquet_path)

                if not set(KEY_COLS).issubset(old.columns):
                    old = pd.DataFrame()

            except Exception:
                old = pd.DataFrame()

            df = (
                pd.concat([old, new], ignore_index=True)
                .drop_duplicates(KEY_COLS, keep="last")
            )
        else:
            df = new

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

    def _process_one(self, extractor, name, item):
        start = time.time()

        row = dict(
            item,
            ocr_extractor=name,
        )

        try:
            text = extractor.extract(
                item["image_path"],
                row,
            )

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

            self.logger.exception(
                "[%s] OCR failed for %s",
                name,
                item["image_path"],
            )

        return row

    def run(self):
        if not self.binarization_parquet.exists():
            raise FileNotFoundError(self.binarization_parquet)

        inputs = self._discover_inputs()
        done = self._load_done()
        processed = 0

        for name in self.extractor_names:

            if name not in OCR_EXTRACTORS:
                raise ValueError(
                    f"Unknown OCR extractor {name!r}; "
                    f"choose from {list(OCR_EXTRACTORS)}"
                )

            pending = []

            for item in inputs:
                key = (
                    tuple(item[c] for c in KEY_COLS[:-1])
                    + (name,)
                )

                if key not in done:
                    pending.append(item)

            if not pending:
                self.logger.info(
                    "Skipping %s: all jobs already completed.",
                    name,
                )
                continue

            self.logger.info(
                "Running %s OCR on %d images using %d workers",
                name,
                len(pending),
                self.workers,
            )

            def process(item):
                extractor = OCR_EXTRACTORS[name]()

                try:
                    return self._process_one(
                        extractor,
                        name,
                        item,
                    )
                finally:
                    extractor.close()

            with ThreadPoolExecutor(
                max_workers=self.workers
            ) as executor:

                futures = [
                    executor.submit(process, item)
                    for item in pending
                ]

                for future in as_completed(futures):
                    row = future.result()

                    self._append(row)
                    processed += 1

                    if row["status"] == "success":
                        key = tuple(
                            row[c] for c in KEY_COLS
                        )
                        done.add(key)

        return processed