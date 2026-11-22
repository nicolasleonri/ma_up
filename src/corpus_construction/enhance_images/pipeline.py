"""Image preprocessing pipeline with multi-process parallelism."""
from concurrent.futures import ProcessPoolExecutor, as_completed
import cv2
import logging
import os
import pandas as pd
from pathlib import Path
import time
from .steps import (
    Contrast,
    Denoising,
    Sharpening,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = ".enhancement_checkpoint.txt"

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
}

NEWSPAPER_CONFIG_IDS: dict[str, list[int]] = {
    "correo": [13],
    "elcomercio": [51],
    "gestion": [6],
    "ojo": [52],
    "peru21": [38],
    "publimetro": [12],
    "trome": [34],  # Default for non-last pages; last pages auto-route to [7]
}

CONTRAST_METHODS = {
    "none": Contrast.none,
    "clahe": Contrast.clahe,
}

DENOISING_METHODS = {
    "none": Denoising.none,
    "mean_filter": Denoising.mean_filter,
    "gaussian_filter": Denoising.gaussian_filter,
    "median_filter": Denoising.median_filter,
    "conservative_filter": Denoising.conservative_filter,
    "laplacian_filter": Denoising.laplacian_filter,
    "frequency_filter": Denoising.frequency_filter,
    "crimmins_speckle_removal": Denoising.crimmins_speckle_removal,
    "unsharp_filter": Denoising.unsharp_filter,
}

SHARPENING_METHODS = {
    "none": Sharpening.none,
    "unsharp_masking": Sharpening.unsharp_mask,
    "stroke_width_enhancement": Sharpening.stroke_width_enhancement,
}

def _extract_edition_and_page(stem: str) -> tuple[str, int] | None:
    """Parse 'trome_2014-10-07_32' -> ('trome_2014-10-07', 32)."""
    parts = stem.rsplit("_", 1)
    if len(parts) == 2 and parts[1].isdigit():
        return parts[0], int(parts[1])
    return None

def _find_last_page_stems(image_files: list[Path]) -> set[str]:
    """Identify the stem with the highest page number per (newspaper_date) edition."""
    max_page_by_edition: dict[str, tuple[int, str]] = {}
    for img in image_files:
        parsed = _extract_edition_and_page(img.stem)
        if parsed is None:
            continue
        edition, page_num = parsed
        current = max_page_by_edition.get(edition)
        if current is None or page_num > current[0]:
            max_page_by_edition[edition] = (page_num, img.stem)
    return {stem for _, stem in max_page_by_edition.values()}

def _worker_init() -> None:
    """Prevent OpenCV thread oversubscription inside worker processes."""
    cv2.setNumThreads(1)


def _process_single_image_task(
    image_path_str: str,
    relative_path: str,
    output_dir_str: str,
    active_configs: list[dict],
) -> tuple[str, Path, list[dict]]:
    """Top-level picklable worker function to process and save one image."""
    image_path = Path(image_path_str)
    output_dir = Path(output_dir_str)
    output_dir.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        return relative_path, output_dir, []

    metadata_records: list[dict] = []

    for config in active_configs:
        start = time.time()
        cid = config["config_id"]
        contrast_name = config["contrast"]
        denoising_name = config["denoising"]
        sharpening_name = config["sharpening"]

        try:
            img = image.copy()
            img = CONTRAST_METHODS[contrast_name](img)
            img = DENOISING_METHODS[denoising_name](img)
            img = SHARPENING_METHODS[sharpening_name](img)

            output_filename = f"{image_path.stem}_config_{cid:03d}.tiff"
            output_file = output_dir / output_filename
            cv2.imwrite(str(output_file), img)

            elapsed = time.time() - start
            output_relative = str(Path(relative_path).parent / output_filename)

            metadata_records.append(
                {
                    "image_path": relative_path,
                    "config_id": cid,
                    "contrast": contrast_name,
                    "denoising": denoising_name,
                    "sharpening": sharpening_name,
                    "output_path": output_relative,
                    "processing_time_seconds": round(elapsed, 3),
                    "status": "success",
                }
            )
        except Exception as exc:
            metadata_records.append(
                {
                    "image_path": relative_path,
                    "config_id": cid,
                    "contrast": contrast_name,
                    "denoising": denoising_name,
                    "sharpening": sharpening_name,
                    "output_path": None,
                    "processing_time_seconds": None,
                    "status": f"failed: {exc}",
                }
            )

    return relative_path, output_dir, metadata_records


class EnhancementPipeline:
    def __init__(self, workers: int = 1):
        self.workers = max(1, workers)
        self.contrast_methods = CONTRAST_METHODS
        self.denoising_methods = DENOISING_METHODS
        self.sharpening_methods = SHARPENING_METHODS

    def _build_configs(self) -> list[dict]:
        """Build all 54 preprocessing configurations with stable config_id."""
        configs = []
        idx = 0
        for contrast in self.contrast_methods:
            for denoising in self.denoising_methods:
                for sharpening in self.sharpening_methods:
                    configs.append(
                        {
                            "config_id": idx,
                            "contrast": contrast,
                            "denoising": denoising,
                            "sharpening": sharpening,
                        }
                    )
                    idx += 1
        assert len(configs) == 54
        return configs

    def _select_configs(
        self,
        all_configs: list[dict],
        config_ids: list[int] | None,
    ) -> list[dict]:
        """Filter the 54 configurations to the requested config_ids."""
        if not config_ids:
            return all_configs

        valid_ids = {c["config_id"] for c in all_configs}
        unknown = sorted(set(config_ids) - valid_ids)
        if unknown:
            raise ValueError(
                f"Invalid config_id(s): {unknown}. Must be in 0..53."
            )

        selected_set = set(config_ids)
        return [c for c in all_configs if c["config_id"] in selected_set]

    def save_configurations(
        self,
        configs: list[dict],
        output_dir: Path,
    ):
        config_file = output_dir / "configurations.txt"
        with open(config_file, "w") as f:
            f.write("IMAGE PREPROCESSING CONFIGURATIONS\n")
            f.write("=================================\n\n")
            f.write(f"Total configurations: {len(configs)}\n")
            f.write("Pipeline order: Contrast -> Denoising -> Sharpening\n\n")

            for config in configs:
                cid = config["config_id"]
                f.write(f"config_{cid:03d}\n")
                f.write(f"  contrast:    {config['contrast']}\n")
                f.write(f"  denoising:   {config['denoising']}\n")
                f.write(f"  sharpening:  {config['sharpening']}\n\n")

    def _flush_metadata_buffers(
        self,
        buffers: dict[Path, list[dict]],
    ) -> None:
        """Flush buffered metadata records per output_dir in the main process."""
        for out_dir, records in list(buffers.items()):
            if not records:
                continue
            parquet_path = out_dir / "enhance_images.parquet"
            df_new = pd.DataFrame(records)

            if parquet_path.exists():
                df_existing = pd.read_parquet(parquet_path)
                df_combined = pd.concat(
                    [df_existing, df_new],
                    ignore_index=True,
                ).drop_duplicates(
                    subset=["image_path", "config_id"],
                    keep="last",
                )
            else:
                df_combined = df_new

            df_combined.to_parquet(parquet_path, index=False)
            buffers[out_dir] = []

    def _load_checkpoint(self, checkpoint_path: Path) -> set[str]:
        if not checkpoint_path.exists():
            return set()
        with open(checkpoint_path, "r") as f:
            return {line.strip() for line in f if line.strip()}

    def run(
        self,
        input_root: Path,
        output_root: Path,
        resume: bool = True,
        config_ids: list[int] | None = None,
        sub_dirs: list[str] | None = None,
        newspaper: str | None = None,
    ) -> int:
        """Run selected preprocessing configurations across CPU workers."""
        output_root.mkdir(parents=True, exist_ok=True)

        all_configs = self._build_configs()
        self.save_configurations(all_configs, output_root)
        configs_by_id = {c["config_id"]: c for c in all_configs}

        use_trome_split = (
            newspaper is not None
            and newspaper.lower() == "trome"
            and config_ids is None
        )

        if use_trome_split:
            logger.info(
                "Trome position-aware routing active: last page -> [7], other pages -> [34] | Workers: %d",
                self.workers,
            )
        else:
            default_active = self._select_configs(all_configs, config_ids)
            logger.info(
                "Active config_id(s): %s | Workers: %d",
                [c["config_id"] for c in default_active],
                self.workers,
            )

        ckpt_name = (
            f".enhancement_checkpoint_{'_'.join(sub_dirs)}.txt"
            if sub_dirs
            else CHECKPOINT_FILENAME
        )
        checkpoint_path = output_root / ckpt_name
        done = self._load_checkpoint(checkpoint_path) if resume else set()

        scan_roots = (
            [input_root / sd for sd in sub_dirs if (input_root / sd).exists()]
            if sub_dirs
            else [input_root]
        )

        image_files = sorted(
            f
            for root in scan_roots
            for f in root.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        last_page_stems = (
            _find_last_page_stems(image_files) if use_trome_split else set()
        )

        total = len(image_files)
        pending_tasks: list[tuple[str, str, str, list[dict], str]] = []
        skipped = 0

        for image_path in image_files:
            relative = str(image_path.relative_to(input_root))

            if use_trome_split:
                img_cids = [7] if image_path.stem in last_page_stems else [34]
                img_configs = [configs_by_id[cid] for cid in img_cids]
                ids_tag = ",".join(str(cid) for cid in img_cids)
                checkpoint_key = f"{relative}::{ids_tag}"
            else:
                img_configs = self._select_configs(all_configs, config_ids)
                ids_tag = ",".join(str(c["config_id"]) for c in img_configs)
                checkpoint_key = (
                    f"{relative}::{ids_tag}" if config_ids else relative
                )

            if resume and (checkpoint_key in done or relative in done):
                skipped += 1
                continue

            out_dir = output_root / image_path.relative_to(input_root).parent
            pending_tasks.append(
                (str(image_path), relative, str(out_dir), img_configs, checkpoint_key)
            )

        logger.info(
            "Found %d image(s) under %s (skipped=%d, pending=%d)",
            total,
            input_root,
            skipped,
            len(pending_tasks),
        )

        if not pending_tasks:
            return 0

        processed = 0
        metadata_buffers: dict[Path, list[dict]] = {}

        with open(checkpoint_path, "a") as ckpt_file:
            if self.workers == 1:
                _worker_init()
                for img_str, rel, out_str, img_configs, ckpt_key in pending_tasks:
                    rel_done, out_dir, records = _process_single_image_task(
                        img_str, rel, out_str, img_configs
                    )
                    if not records:
                        logger.warning("Could not read %s, skipping.", img_str)
                        continue

                    metadata_buffers.setdefault(out_dir, []).extend(records)
                    ckpt_file.write(ckpt_key + "\n")
                    processed += 1

                    if processed % 50 == 0:
                        self._flush_metadata_buffers(metadata_buffers)
                        ckpt_file.flush()
                        logger.info(
                            "Progress: %d/%d pending images completed",
                            processed,
                            len(pending_tasks),
                        )
            else:
                with ProcessPoolExecutor(
                    max_workers=self.workers,
                    initializer=_worker_init,
                ) as pool:
                    futures = {
                        pool.submit(
                            _process_single_image_task,
                            img_str,
                            rel,
                            out_str,
                            img_configs,
                        ): (rel, ckpt_key)
                        for img_str, rel, out_str, img_configs, ckpt_key in pending_tasks
                    }

                    for future in as_completed(futures):
                        rel, ckpt_key = futures[future]
                        try:
                            rel_done, out_dir, records = future.result()
                        except Exception as exc:
                            logger.exception("Worker failed on %s: %s", rel, exc)
                            continue

                        if not records:
                            logger.warning("Could not read %s, skipping.", rel)
                            continue

                        metadata_buffers.setdefault(out_dir, []).extend(records)
                        ckpt_file.write(ckpt_key + "\n")
                        processed += 1

                        if processed % 50 == 0:
                            self._flush_metadata_buffers(metadata_buffers)
                            ckpt_file.flush()
                            logger.info(
                                "Progress: %d/%d pending images completed",
                                processed,
                                len(pending_tasks),
                            )

        self._flush_metadata_buffers(metadata_buffers)

        logger.info(
            "Enhancement complete. Processed: %d, Skipped: %d, Total: %d",
            processed,
            skipped,
            total,
        )
        return processed