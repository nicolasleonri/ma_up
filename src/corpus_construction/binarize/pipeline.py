import logging
import os
import re
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import cv2
import pandas as pd

from .steps import Binarization

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}

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

FULL_PAGE_NEWSPAPERS = {"elcomercio", "gestion"}

NEWSPAPER_BINARIZATION: Dict[str, List[str]] = {
    "correo": ["basic"],
    "elcomercio": ["none_color"],
    "gestion": ["otsu"],
    "ojo": ["basic"],
    "peru21": ["none_color"],
    "publimetro": ["none_grayscale"],
    "trome": ["otsu", "none_color"],
}

TROME_DETECTOR_BINARIZATION = {
    "doclayout": "otsu",
    "ppdoclayout": "none_color",
}

METADATA_COLUMNS = [
    "image_stem",
    "detector",
    "config_id",
    "binarization",
    "binarize_file",
    "elapsed_s",
    "status",
]


def _binarize_single_worker(
    args: Tuple[str, str, List[str], Optional[str]]
) -> Tuple[str, List[Dict], List[Dict], bool]:
    """Worker function for ProcessPoolExecutor."""
    image_path_str, relative_path, methods, newspaper = args
    image_path = Path(image_path_str)

    image = cv2.imread(image_path_str, cv2.IMREAD_UNCHANGED)
    if image is None:
        return relative_path, [], [], False

    stem = image_path.stem
    match = re.match(
        r"^(?P<image_stem>.+?)_config_(?P<config_id>\d+)(?P<suffix>.*)$",
        stem,
    )
    if match is None:
        return relative_path, [], [], False

    image_stem = match.group("image_stem")
    config_id = int(match.group("config_id"))
    suffix = match.group("suffix")
    detector = None
    if suffix:
        parts = [p for p in suffix.split("_") if p]
        if parts:
            detector = parts[0]

    # Route trome by detector: doclayout -> otsu, ppdoclayout -> none_color
    active_methods = methods
    if newspaper == "trome" and detector in TROME_DETECTOR_BINARIZATION:
        active_methods = [TROME_DETECTOR_BINARIZATION[detector]]

    outputs: List[Dict] = []
    records: List[Dict] = []
    all_ok = True

    for method_name in active_methods:
        bin_idx = BINARIZATION_METHODS.index(method_name)
        start = time.perf_counter()
        out_name = f"{stem}_bin_{bin_idx:03d}.tiff"
        out_rel = str(Path(relative_path).parent / out_name)
        try:
            img_out = getattr(Binarization, method_name)(image.copy())
            elapsed = round(time.perf_counter() - start, 3)
            outputs.append(
                {
                    "image": img_out,
                    "filename": out_name,
                    "relative_parent": str(Path(relative_path).parent),
                }
            )
            records.append(
                {
                    "image_stem": image_stem,
                    "detector": detector,
                    "config_id": config_id,
                    "binarization": method_name,
                    "binarize_file": out_rel,
                    "elapsed_s": elapsed,
                    "status": "success",
                }
            )
        except Exception as exc:
            all_ok = False
            elapsed = round(time.perf_counter() - start, 3)
            records.append(
                {
                    "image_stem": image_stem,
                    "detector": detector,
                    "config_id": config_id,
                    "binarization": method_name,
                    "binarize_file": None,
                    "elapsed_s": elapsed,
                    "status": f"failed: {exc}",
                }
            )

    return relative_path, outputs, records, all_ok


class BinarizationPipeline:
    """Apply newspaper-specific or configured binarization methods."""

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
        self.methods = list(methods) if methods is not None else list(BINARIZATION_METHODS)
        self.workers = max(1, int(workers))
        self.newspaper = newspaper.lower() if newspaper else None

    @staticmethod
    def _load_checkpoint(checkpoint_path: Path) -> Set[str]:
        if not checkpoint_path.exists():
            return set()
        with checkpoint_path.open("r", encoding="utf-8") as f:
            return {line.strip() for line in f if line.strip()}

    @staticmethod
    def _write_outputs(outputs: List[Dict], output_root: Path) -> bool:
        for result in outputs:
            out_dir = output_root / result["relative_parent"]
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / result["filename"]
            if not cv2.imwrite(str(out_path), result["image"]):
                return False
        return True

    def _save_metadata(self, records: List[Dict], parquet_path: Path) -> None:
        if not records:
            return
        df_new = pd.DataFrame(records)
        if parquet_path.exists():
            df_existing = pd.read_parquet(parquet_path)
            df_combined = pd.concat([df_existing, df_new], ignore_index=True)
        else:
            df_combined = df_new

        unique_cols = [
            "image_stem",
            "detector",
            "config_id",
            "binarization",
            "binarize_file",
        ]
        df_combined = (
            df_combined.drop_duplicates(subset=unique_cols, keep="last")
            .reset_index(drop=True)[METADATA_COLUMNS]
        )
        df_combined.to_parquet(parquet_path, index=False)
        self.logger.info(
            "Saved binarization metadata: %d records -> %s",
            len(df_combined),
            parquet_path,
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

        active_newspaper = newspaper.lower() if newspaper else self.newspaper
        output_root.mkdir(parents=True, exist_ok=True)

        shard_tag = ""
        if sub_dirs:
            sanitized = "_".join(str(s).strip("/").replace("/", "_") for s in sub_dirs)
            shard_tag = f"_{sanitized}"

        checkpoint_path = output_root / f".binarization_checkpoint{shard_tag}.txt"
        parquet_path = output_root / f"binarization{shard_tag}.parquet"

        done = self._load_checkpoint(checkpoint_path) if resume else set()

        search_roots = (
            [input_root / sd for sd in sub_dirs if (input_root / sd).exists()]
            if sub_dirs
            else [input_root]
        )

        image_files: List[Path] = []
        for root in search_roots:
            image_files.extend(
                p
                for p in root.rglob("*")
                if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
            )
        image_files.sort()

        pending_tasks: List[Tuple[str, str, List[str], Optional[str]]] = []
        skipped = 0
        for img_path in image_files:
            rel_path = str(img_path.relative_to(input_root))
            if resume and rel_path in done:
                skipped += 1
                continue
            pending_tasks.append((str(img_path), rel_path, self.methods, active_newspaper))

        total = len(image_files)
        self.logger.info(
            "Found %d image(s) (pending=%d, skipped=%d, workers=%d, methods=%s)",
            total,
            len(pending_tasks),
            skipped,
            self.workers,
            self.methods,
        )

        if not pending_tasks:
            return 0

        processed = 0
        failed = 0
        all_records: List[Dict] = []

        with (
            ProcessPoolExecutor(max_workers=self.workers) as executor,
            checkpoint_path.open("a", encoding="utf-8") as ckpt_file,
        ):
            for idx, (rel_path, outputs, records, all_ok) in enumerate(
                executor.map(_binarize_single_worker, pending_tasks, chunksize=8),
                start=1,
            ):
                all_records.extend(records)
                if all_ok and self._write_outputs(outputs, output_root):
                    ckpt_file.write(rel_path + "\n")
                    processed += 1
                else:
                    failed += 1

                if idx % 250 == 0:
                    ckpt_file.flush()
                    self.logger.info(
                        "Progress: %d/%d pending images completed",
                        idx,
                        len(pending_tasks),
                    )

        self._save_metadata(all_records, parquet_path)
        self.logger.info(
            "Binarization complete. Processed: %d, Skipped: %d, Failed: %d, Total: %d",
            processed,
            skipped,
            failed,
            total,
        )
        return processed
