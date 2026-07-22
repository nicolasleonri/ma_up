"""Image preprocessing pipeline.

Applies all combinations of:

    Contrast:     2 techniques
    Denoising:    9 techniques
    Sharpening:   3 techniques

Total:

    2 x 9 x 3 = 54 preprocessing configurations

Each configuration is applied sequentially:

    Contrast -> Denoising -> Sharpening
"""
import cv2
import time
import logging
import pandas as pd
from pathlib import Path
from .steps import (
    Contrast,
    Denoising,
    Sharpening,
)

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = (
    ".enhancement_checkpoint.txt"
)

SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".tiff",
    ".tif",
}

class EnhancementPipeline:
    def __init__(self):
        self.contrast_methods = {
            "none": Contrast.none,
            "clahe": Contrast.clahe,
        }
        self.denoising_methods = {
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
        self.sharpening_methods = {
            "none": Sharpening.none,
            "unsharp_masking": Sharpening.unsharp_mask,
            "stroke_width_enhancement": Sharpening.stroke_width_enhancement,
        }

    def _build_configs(self):
        """Build all 54 preprocessing configurations."""
        configs = []
        for contrast in self.contrast_methods:
            for denoising in self.denoising_methods:
                for sharpening in self.sharpening_methods:
                    configs.append(
                        {
                            "contrast": contrast,
                            "denoising": denoising,
                            "sharpening": sharpening,
                        }
                    )
        assert len(configs) == 54
        return configs

    def save_configurations(
        self,
        configs,
        output_dir: Path,
    ):
        config_file = (
            output_dir
            / "configurations.txt"
        )

        with open(config_file, "w") as f:
            f.write("IMAGE PREPROCESSING CONFIGURATIONS\n")
            f.write("=================================\n\n")
            f.write("Total configurations: " f"{len(configs)}\n")
            f.write("Pipeline order: Contrast -> Denoising -> Sharpening\n\n")

            for idx, config in enumerate(configs):
                f.write(f"config_{idx:03d}\n")
                f.write(
                    f"  contrast:    "
                    f"{config['contrast']}\n"
                )

                f.write(
                    f"  denoising:   "
                    f"{config['denoising']}\n"
                )

                f.write(
                    f"  sharpening:  "
                    f"{config['sharpening']}\n"
                )

                f.write("\n")

        logger.info(
            "Saved %d configurations to %s",
            len(configs),
            config_file,
        )

    def _save_metadata_parquet(
        self,
        metadata_records: list[dict],
        output_dir: Path,
    ):
        """Append metadata records to the parquet file."""
        parquet_path = output_dir / "enhance_images.parquet"

        df_new = pd.DataFrame(metadata_records)

        if parquet_path.exists():
            df_existing = pd.read_parquet(parquet_path)
            df_combined = pd.concat(
                [df_existing, df_new],
                ignore_index=True,
            )
        else:
            df_combined = df_new

        df_combined.to_parquet(
            parquet_path,
            index=False,
        )

        logger.info(
            "Saved %d metadata records to %s",
            len(df_combined),
            parquet_path,
        )

    def process(
        self,
        image,
        image_name: str,
        configs,
        relative_path: str,
        output_dir: Path,
    ):
        """Apply all 54 configurations to one image."""

        outputs = []
        metadata_records = []

        for idx, config in enumerate(configs):
            start = time.time()

            contrast_name = config["contrast"]
            denoising_name = config["denoising"]
            sharpening_name = config["sharpening"]

            logger.info(
                "[%s] Config %d/%d | "
                "contrast=%s | "
                "denoising=%s | "
                "sharpening=%s",
                image_name,
                idx + 1,
                len(configs),
                contrast_name,
                denoising_name,
                sharpening_name,
            )

            try:
                img = image.copy()
                img = self.contrast_methods[contrast_name](img) 
                img = self.denoising_methods[denoising_name](img)
                img = self.sharpening_methods[sharpening_name](img)

                elapsed = (time.time() - start)

                output_filename = (
                    f"{Path(image_name).stem}"
                    f"_config_"
                    f"{idx:03d}"
                    ".tiff"
                )
                output_relative = str(
                    Path(relative_path).parent / output_filename
                )

                outputs.append(
                    {
                        "config": idx,
                        "contrast":
                            contrast_name,
                        "denoising":
                            denoising_name,
                        "sharpening":
                            sharpening_name,
                        "image": img,
                        "time": elapsed,
                    }
                )

                # Collect metadata for parquet
                metadata_records.append(
                    {
                        "image_path": relative_path,
                        "config_id": idx,
                        "contrast": contrast_name,
                        "denoising": denoising_name,
                        "sharpening": sharpening_name,
                        "output_path": output_relative,
                        "processing_time_seconds": round(elapsed, 3),
                        "status": "success",
                    }
                )

                logger.info(
                    "[%s] Config %d finished "
                    "in %.3fs",
                    image_name,
                    idx,
                    elapsed,
                )

            except Exception as exc:
                logger.exception(
                    "[%s] FAILED config %d: %s",
                    image_name,
                    idx,
                    exc,
                )

                # Record failure in metadata
                metadata_records.append(
                    {
                        "image_path": relative_path,
                        "config_id": idx,
                        "contrast": contrast_name,
                        "denoising": denoising_name,
                        "sharpening": sharpening_name,
                        "output_path": None,
                        "processing_time_seconds": None,
                        "status": f"failed: {exc}",
                    }
                )

        if metadata_records:
            self._save_metadata_parquet(
                metadata_records,
                output_dir,
            )

        return outputs

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ):
        if not checkpoint_path.exists():
            return set()

        with open(
            checkpoint_path,
            "r",
        ) as f:

            return {
                line.strip()
                for line in f
                if line.strip()
            }

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        relative_path: str,
    ):
        with open(
            checkpoint_path,
            "a",
        ) as f:

            f.write(
                relative_path
                + "\n"
            )

    def run(
        self,
        input_root: Path,
        output_root: Path,
        resume: bool = True,
    ):
        """Run the 54 preprocessing configurations."""

        output_root.mkdir(
            parents=True,
            exist_ok=True,
        )

        checkpoint_path = (
            output_root
            / CHECKPOINT_FILENAME
        )

        done = (
            self._load_checkpoint(
                checkpoint_path
            )
            if resume
            else set()
        )

        configs = (
            self._build_configs()
        )

        self.save_configurations(
            configs,
            output_root,
        )

        image_files = sorted(
            f
            for f in input_root.rglob("*")
            if (
                f.is_file()
                and f.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

        total = len(
            image_files
        )

        processed = 0
        skipped = 0

        logger.info(
            "Found %d image(s) under %s",
            total,
            input_root,
        )

        for image_path in image_files:

            relative = str(
                image_path.relative_to(
                    input_root
                )
            )

            if (
                resume
                and relative in done
            ):
                skipped += 1
                continue

            output_dir = (
                output_root
                / image_path.relative_to(
                    input_root
                ).parent
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            image = cv2.imread(
                str(image_path)
            )

            if image is None:
                logger.warning(
                    "Could not read %s, "
                    "skipping.",
                    image_path,
                )
                continue

            results = self.process(
                image,
                image_path.name,
                configs,
                relative,
                output_dir
            )

            for result in results:

                output_file = (
                    output_dir
                    / (
                        f"{image_path.stem}"
                        f"_config_"
                        f"{result['config']:03d}"
                        ".tiff"
                    )
                )

                cv2.imwrite(
                    str(output_file),
                    result["image"],
                )

            self._save_checkpoint(
                checkpoint_path,
                relative,
            )

            processed += 1

        logger.info(
            "Enhancement complete. "
            "Processed: %d, "
            "Skipped: %d, "
            "Total: %d",
            processed,
            skipped,
            total,
        )

        return processed
