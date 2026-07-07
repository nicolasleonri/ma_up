"""Enhancement pipeline: walks raw images, applies all enhancement combinations, saves results.

Generates every combination of CLAHE / denoising / unsharp masking steps,
mirroring the structure of ``ImagePreprocessingPipeline``. Each combination
is saved as a separate TIFF alongside a ``configurations.txt`` index.

Input:  data/raw/images/{newspaper}/{Y}/{M}/{D}/{filename}.jpg
Output: data/processed/enhanced/{newspaper}/{Y}/{M}/{D}/{filename}_config_{idx}.tiff

Checkpointing via a plain-text log file so interrupted runs resume without
reprocessing completed images.
"""

import time
import logging
from pathlib import Path

import cv2

from .steps import Enhancement

logger = logging.getLogger(__name__)

CHECKPOINT_FILENAME = ".enhancement_checkpoint.txt"
SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".tiff", ".tif"}


class EnhancementPipeline:

    def __init__(self):
        pass

    def _build_configs(self) -> list[tuple[str, str, str]]:
        """Return all (clahe, denoise, unsharp) step combinations."""
        configs = []
        for c in ["none", "clahe"]:
            for d in ["none", "denoise"]:
                for u in ["none", "unsharp_mask"]:
                    configs.append((c, d, u))
        return configs

    def save_configurations(self, configs: list[tuple], output_dir: Path) -> None:
        config_file = output_dir / "configurations.txt"
        with open(config_file, "w") as f:
            for idx, (c, d, u) in enumerate(configs):
                f.write(f"config_{idx}\n")
                f.write(f"  clahe:        {c}\n")
                f.write(f"  denoise:      {d}\n")
                f.write(f"  unsharp_mask: {u}\n")
                f.write("\n")
        logger.info("Saved enhancement configurations to %s", config_file)

    def process(self, image, image_name: str, configs: list[tuple]) -> list[dict]:
        """Apply every combination to a single image, return list of result dicts."""
        outputs = []

        for idx, (clahe_step, denoise_step, unsharp_step) in enumerate(configs):
            start = time.time()
            logger.info(
                "[%s] Config %d/%d | clahe=%s | denoise=%s | unsharp_mask=%s",
                image_name, idx + 1, len(configs), clahe_step, denoise_step, unsharp_step,
            )

            try:
                img = image.copy()

                if clahe_step != "none":
                    img = Enhancement.clahe(img)
                if denoise_step != "none":
                    img = Enhancement.denoise(img)
                if unsharp_step != "none":
                    img = Enhancement.unsharp_mask(img)

                elapsed = time.time() - start
                logger.info("[%s] Config %d finished in %.3fs", image_name, idx, elapsed)

                outputs.append({
                    "config": idx,
                    "clahe": clahe_step,
                    "denoise": denoise_step,
                    "unsharp_mask": unsharp_step,
                    "image": img,
                    "time": elapsed,
                })

            except Exception as exc:
                logger.exception("[%s] FAILED config %d: %s", image_name, idx, exc)

        return outputs

    def _load_checkpoint(self, checkpoint_path: Path) -> set[str]:
        if not checkpoint_path.exists():
            return set()
        with open(checkpoint_path, "r") as f:
            return {line.strip() for line in f if line.strip()}

    def _save_checkpoint(self, checkpoint_path: Path, relative_path: str) -> None:
        with open(checkpoint_path, "a") as f:
            f.write(relative_path + "\n")

    def run(
        self,
        input_root: Path,
        output_root: Path,
        resume: bool = True,
    ) -> int:
        output_root.mkdir(parents=True, exist_ok=True)
        checkpoint_path = output_root / CHECKPOINT_FILENAME
        done = self._load_checkpoint(checkpoint_path) if resume else set()

        configs = self._build_configs()
        self.save_configurations(configs, output_root)

        image_files = sorted(
            f for f in input_root.rglob("*")
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
        )

        total = len(image_files)
        processed = 0
        skipped = 0

        logger.info("Found %d image(s) under %s", total, input_root)

        for image_path in image_files:
            relative = str(image_path.relative_to(input_root))

            if resume and relative in done:
                skipped += 1
                continue

            output_dir = (output_root / image_path.relative_to(input_root)).parent
            output_dir.mkdir(parents=True, exist_ok=True)

            image = cv2.imread(str(image_path))
            if image is None:
                logger.warning("Could not read %s, skipping.", image_path)
                continue

            results = self.process(image, image_path.name, configs)

            for result in results:
                out = output_dir / f"{image_path.stem}_config_{result['config']}.tiff"
                cv2.imwrite(str(out), result["image"])

            self._save_checkpoint(checkpoint_path, relative)
            processed += 1

        logger.info(
            "Enhancement complete. Processed: %d, Skipped (already done): %d, Total: %d",
            processed, skipped, total,
        )
        return processed