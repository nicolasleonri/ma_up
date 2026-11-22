"""Binarization pipeline with newspaper/position routing, shard safety, and CPU parallelism."""

from concurrent.futures import ProcessPoolExecutor, as_completed
import logging
import os
import re
import time
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import pandas as pd

from .steps import Binarization

CHECKPOINT_FILENAME = ".binarization_checkpoint.txt"
METADATA_FILENAME = "binarization.parquet"
CONFIGURATION_FILENAME = "configurations.txt"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
}

BINARIZATION_METHODS = [
    "none_color",
    "none_grayscale",
    "basic",
    "otsu",
    "adaptive_mean",
    "adaptive_gaussian",
    "yannihorne",
    "niblack",
]

METHOD_INDEX_MAP: Dict[str, int] = {
    name: idx for idx, name in enumerate(BINARIZATION_METHODS)
}

# Winning binarization method(s) per newspaper
NEWSPAPER_BINARIZATION: Dict[str, List[str]] = {
    "correo": ["basic"],
    "elcomercio": ["none_color"],
    "gestion": ["otsu"],
    "ojo": ["basic"],
    "peru21": ["none_color"],
    "publimetro": ["none_grayscale"],
    "trome": ["otsu", "none_color"],
}

# Position-specific config_id -> binarization method routing for Trome
TROME_CONFIG_BINARIZATION: Dict[int, str] = {
    7: "otsu",         # Last page
    34: "none_color",  # Interior / other pages
}

# Newspapers that bypass layout cropping (read directly from enhanced/)
FULL_PAGE_NEWSPAPERS = {"elcomercio", "gestion"}

CONFIG_ID_PATTERN = re.compile(r"_config_(\d+)(?:_|$)")


def _worker_init() -> None:
    """Prevent OpenCV thread oversubscription inside worker processes."""
    cv2.setNumThreads(1)


def _binarize_single_image_task(
    image_path_str: str,
    relative_path: str,
    output_root_str: str,
    image_stem: str,
    detector: Optional[str],
    config_id: int,
    methods_to_run: List[str],
) -> Tuple[str, List[Dict], bool]:
    """Top-level picklable worker function to binarize one image and write outputs."""
    image_path = Path(image_path_str)
    output_root = Path(output_root_str)

    image = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    if image is None:
        return relative_path, [], False

    records: List[Dict] = []
    all_successful = True
    rel_parent = Path(relative_path).parent
    out_dir = output_root / rel_parent
    out_dir.mkdir(parents=True, exist_ok=True)

    for binarization_name in methods_to_run:
        bin_idx = METHOD_INDEX_MAP[binarization_name]
        output_filename = f"{image_path.stem}_bin_{bin_idx:03d}.tiff"
        output_relative = str(rel_parent / output_filename)
        start = time.perf_counter()

        try:
            img = image.copy()
            method = getattr(Binarization, binarization_name)
            img = method(img)

            out_path = out_dir / output_filename
            if not cv2.imwrite(str(out_path), img):
                raise IOError(f"cv2.imwrite failed for {out_path}")

            elapsed = time.perf_counter() - start
            records.append(
                {
                    "image_stem": image_stem,
                    "detector": detector,
                    "config_id": config_id,
                    "binarization": binarization_name,
                    "binarize_file": output_relative,
                    "elapsed_s": round(elapsed, 3),
                    "status": "success",
                }
            )
        except Exception as exc:
            all_successful = False
            elapsed = time.perf_counter() - start
            records.append(
                {
                    "image_stem": image_stem,
                    "detector": detector,
                    "config_id": config_id,
                    "binarization": binarization_name,
                    "binarize_file": None,
                    "elapsed_s": round(elapsed, 3),
                    "status": f"failed: {exc}",
                }
            )

    return relative_path, records, all_successful


class BinarizationPipeline:
    """Apply selected or all binarization methods to enhanced/cropped images."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        layout_parquet: Optional[Path] = None,
        methods: Optional[List[str]] = None,
        workers: int = 1,
        newspaper: Optional[str] = None,
    ) -> None:
        self.logger = logger or logging.getLogger(__name__)
        self.layout_parquet = layout_parquet
        self.workers = max(1, workers)
        self.newspaper = newspaper.lower() if newspaper else None

        if methods is not None:
            unknown = sorted(set(methods) - set(BINARIZATION_METHODS))
            if unknown:
                raise ValueError(f"Unknown binarization method(s): {unknown}")
            self.active_methods = list(methods)
        else:
            self.active_methods = list(BINARIZATION_METHODS)

        self.configs = [{"binarization": m} for m in BINARIZATION_METHODS]

    def _methods_for_variant(
        self,
        newspaper: Optional[str],
        config_id: int,
    ) -> List[str]:
        """Resolve which binarization method(s) apply to a given (newspaper, config_id)."""
        if (
            newspaper
            and newspaper.lower() == "trome"
            and config_id in TROME_CONFIG_BINARIZATION
        ):
            target = TROME_CONFIG_BINARIZATION[config_id]
            return [m for m in self.active_methods if m == target]
        return list(self.active_methods)

    @staticmethod
    def _resolve_image_metadata(
        image_name: str,
    ) -> Tuple[str, Optional[str], int]:
        """Extract (image_stem, detector, config_id) from filename."""
        filename_stem = Path(image_name).stem
        match = re.match(
            r"^(?P<image_stem>.+?)_config_(?P<config_id>\d+)(?P<suffix>.*)$",
            filename_stem,
        )
        if match is None:
            raise ValueError(
                f"Could not resolve image metadata from filename: {image_name}"
            )

        image_stem = match.group("image_stem")
        config_id = int(match.group("config_id"))
        suffix = match.group("suffix")

        detector = None
        if suffix:
            suffix_parts = [part for part in suffix.split("_") if part]
            if suffix_parts:
                detector = suffix_parts[0]

        return image_stem, detector, config_id

    def save_configurations(self, output_dir: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)
        config_file = output_dir / CONFIGURATION_FILENAME
        with config_file.open("w", encoding="utf-8") as file:
            file.write("BINARIZATION CONFIGURATIONS\n")
            file.write("===========================\n\n")
            file.write(f"Total configurations: {len(self.configs)}\n\n")
            for idx, config in enumerate(self.configs):
                file.write(f"binarization_{idx:03d}\n")
                file.write(f"  binarization: {config['binarization']}\n\n")

    def _flush_metadata(
        self,
        records: List[Dict],
        parquet_path: Path,
    ) -> None:
        """Flush buffered metadata records to shard-specific Parquet."""
        if not records:
            return

        df_new = pd.DataFrame(records)
        if parquet_path.exists():
            df_existing = pd.read_parquet(parquet_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        unique_columns = [
            "image_stem",
            "detector",
            "config_id",
            "binarization",
            "binarize_file",
        ]
        columns = [
            "image_stem",
            "detector",
            "config_id",
            "binarization",
            "binarize_file",
            "elapsed_s",
            "status",
        ]

        df_combined = (
            df_combined.drop_duplicates(subset=unique_columns, keep="last")
            .reindex(columns=columns)
            .reset_index(drop=True)
        )
        df_combined.to_parquet(parquet_path, index=False)

    def _load_checkpoint(self, checkpoint_path: Path) -> Set[str]:
        if not checkpoint_path.exists():
            return set()
        with checkpoint_path.open("r", encoding="utf-8") as file:
            return {line.strip() for line in file if line.strip()}

    def _discover_images(
        self,
        input_root: Path,
        sub_dirs: Optional[List[str]],
    ) -> List[Path]:
        """Discover input images, supporting both year subdirs and stem-grouped layout dirs."""
        if not sub_dirs:
            return sorted(
                f
                for f in input_root.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )

        # Case 1: Year subdirectories exist directly under input_root (e.g., enhanced/gestion/2014/)
        existing_sub_paths = [
            input_root / sd for sd in sub_dirs if (input_root / sd).exists()
        ]
        if existing_sub_paths:
            return sorted(
                f
                for root in existing_sub_paths
                for f in root.rglob("*")
                if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
            )

        # Case 2: Layout detection output structure (<newspaper>/<newspaper>_<YYYY>-<MM>-<DD>_<page>/...)
        year_prefixes = tuple(f"_{sd}-" for sd in sub_dirs)
        return sorted(
            f
            for f in input_root.rglob("*")
            if (
                f.is_file()
                and f.suffix.lower() in SUPPORTED_EXTENSIONS
                and any(yp in f.name for yp in year_prefixes)
            )
        )

    def run(
        self,
        input_root: Path,
        output_root: Path,
        resume: bool = True,
        sub_dirs: Optional[List[str]] = None,
        newspaper: Optional[str] = None,
    ) -> int:
        if not input_root.exists():
            raise FileNotFoundError(f"Input directory does not exist: {input_root}")

        active_newspaper = (newspaper or self.newspaper or "").lower() or None
        output_root.mkdir(parents=True, exist_ok=True)
        self.save_configurations(output_root)

        methods_tag = "_".join(sorted(self.active_methods))
        sub_tag = f"_{'_'.join(sub_dirs)}" if sub_dirs else ""
        checkpoint_path = (
            output_root / f".binarization_checkpoint{sub_tag}_{methods_tag}.txt"
        )
        parquet_path = (
            output_root / f"binarization{sub_tag}_{methods_tag}.parquet"
        )

        done = self._load_checkpoint(checkpoint_path) if resume else set()
        image_files = self._discover_images(input_root, sub_dirs)

        total = len(image_files)
        skipped = 0
        pending_tasks: List[
            Tuple[str, str, str, str, Optional[str], int, List[str]]
        ] = []

        for image_path in image_files:
            relative_path = str(image_path.relative_to(input_root))
            if resume and relative_path in done:
                skipped += 1
                continue

            image_stem, detector, config_id = self._resolve_image_metadata(
                image_path.name
            )
            variant_methods = self._methods_for_variant(
                active_newspaper, config_id
            )
            if not variant_methods:
                skipped += 1
                continue

            pending_tasks.append(
                (
                    str(image_path),
                    relative_path,
                    str(output_root),
                    image_stem,
                    detector,
                    config_id,
                    variant_methods,
                )
            )

        self.logger.info(
            "Found %d image(s) under %s (skipped=%d, pending=%d) | newspaper=%s | methods=%s | workers=%d",
            total,
            input_root,
            skipped,
            len(pending_tasks),
            active_newspaper,
            self.active_methods,
            self.workers,
        )

        if not pending_tasks:
            return 0

        processed = 0
        failed = 0
        record_buffer: List[Dict] = []

        with checkpoint_path.open("a", encoding="utf-8") as ckpt_file:
            if self.workers == 1:
                _worker_init()
                for task in pending_tasks:
                    rel_path, records, ok = _binarize_single_image_task(*task)
                    if records:
                        record_buffer.extend(records)
                    if ok:
                        ckpt_file.write(rel_path + "\n")
                        processed += 1
                    else:
                        failed += 1

                    if (processed + failed) % 50 == 0:
                        self._flush_metadata(record_buffer, parquet_path)
                        record_buffer.clear()
                        ckpt_file.flush()
                        self.logger.info(
                            "Progress: %d/%d pending images completed",
                            processed + failed,
                            len(pending_tasks),
                        )
            else:
                with ProcessPoolExecutor(
                    max_workers=self.workers,
                    initializer=_worker_init,
                ) as pool:
                    futures = {
                        pool.submit(_binarize_single_image_task, *task): task[1]
                        for task in pending_tasks
                    }
                    for future in as_completed(futures):
                        rel_path = futures[future]
                        try:
                            _, records, ok = future.result()
                        except Exception as exc:
                            self.logger.exception(
                                "Worker failed on %s: %s", rel_path, exc
                            )
                            failed += 1
                            continue

                        if records:
                            record_buffer.extend(records)
                        if ok:
                            ckpt_file.write(rel_path + "\n")
                            processed += 1
                        else:
                            failed += 1

                        if (processed + failed) % 50 == 0:
                            self._flush_metadata(record_buffer, parquet_path)
                            record_buffer.clear()
                            ckpt_file.flush()
                            self.logger.info(
                                "Progress: %d/%d pending images completed",
                                processed + failed,
                                len(pending_tasks),
                            )

        if record_buffer:
            self._flush_metadata(record_buffer, parquet_path)

        self.logger.info(
            "Binarization complete. Processed: %d, Skipped: %d, Failed: %d, Total: %d",
            processed,
            skipped,
            failed,
            total,
        )
        return processed