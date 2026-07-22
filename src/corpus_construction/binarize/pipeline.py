"""Binarization pipeline.

Applies all 7 binarization methods to enhanced images.

Input images can be:

1. Full-page enhanced images, e.g.

    gestion_2014-01-02_1_config_019.tiff

2. Cropped enhanced images, e.g.

    gestion_2014-01-02_1_config_019_doclayout_article_001.tiff

The enhancement configuration ID is extracted from the input filename:

    gestion_2014-01-02_1_config_019.tiff
                                      ^^^
                                      config_id = 19

The binarization method index is NOT used as config_id.

Instead, the binarization index is used only to generate output
filenames:

    _bin_000.tiff
    _bin_001.tiff
    ...
    _bin_006.tiff

For cropped images, image_stem and detector are resolved from the
layout Parquet.

For full-page images:

    image_stem = parsed from the filename
    detector = None

A single global Parquet is written to:

    <output_dir>/binarization.parquet

Parquet schema
--------------
image_stem       str     canonical page-level image identifier
detector         str     detector name, or null for full-page images
config_id        int     enhancement configuration ID
binarization     str     binarization method name
binarize_file    str     relative path to binarized output
elapsed_s         float  processing time in seconds
status           str     "success" or "failed: <reason>"
"""

import logging
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
    "none",
    "basic",
    "otsu",
    "adaptive_mean",
    "adaptive_gaussian",
    "yannihorne",
    "niblack",
]

LAYOUT_REQUIRED_COLUMNS = {
    "image_stem",
    "detector",
    "crop_file",
}

CONFIG_ID_PATTERN = re.compile(
    r"_config_(\d+)(?:_|$)"
)


class BinarizationPipeline:
    """Apply all binarization methods to enhanced images."""

    def __init__(
        self,
        logger: Optional[logging.Logger] = None,
        layout_parquet: Optional[Path] = None,
    ) -> None:
        self.logger = (
            logger
            or logging.getLogger(__name__)
        )

        self.layout_parquet = layout_parquet

        self.configs = (
            self._build_configs()
        )

        self.layout_metadata = (
            self._load_layout_metadata(
                layout_parquet
            )
            if layout_parquet is not None
            else None
        )

    def _build_configs(
        self,
    ) -> List[Dict[str, str]]:
        """Build the seven binarization configurations."""
        configs = [
            {
                "binarization": method
            }
            for method in BINARIZATION_METHODS
        ]

        if len(configs) != 7:
            raise ValueError(
                "Expected exactly 7 binarization methods, "
                f"found {len(configs)}."
            )

        return configs

    @staticmethod
    def _normalize_path(
        path: str,
    ) -> str:
        """Normalize path separators for reliable matching."""
        return (
            str(path)
            .replace("\\", "/")
            .lstrip("./")
        )

    def _load_layout_metadata(
        self,
        layout_parquet: Path,
    ) -> pd.DataFrame:
        """Load layout metadata used to identify cropped images.

        Expected layout Parquet columns:

            image_stem
            config_id
            detector
            crop_file
            elapsed_s
            status
            error

        Only successful layout records are used.
        """
        if not layout_parquet.exists():
            raise FileNotFoundError(
                "Layout Parquet does not exist: "
                f"{layout_parquet}"
            )

        self.logger.info(
            "Loading layout metadata from %s",
            layout_parquet,
        )

        df = pd.read_parquet(
            layout_parquet
        )

        missing_columns = (
            LAYOUT_REQUIRED_COLUMNS
            - set(df.columns)
        )

        if missing_columns:
            raise ValueError(
                "Layout Parquet is missing required "
                f"columns: {sorted(missing_columns)}"
            )

        if "status" in df.columns:
            df = df[
                df["status"].astype(str)
                == "success"
            ].copy()

        df["crop_file"] = (
            df["crop_file"]
            .astype(str)
            .map(self._normalize_path)
        )

        self.logger.info(
            "Loaded %d successful layout crop "
            "records",
            len(df),
        )

        return df

    @staticmethod
    def _extract_config_id(
        image_name: str,
    ) -> int:
        """Extract enhancement config_id from filename.

        Examples:

            gestion_2014-01-02_1_config_019.tiff
                -> 19

            gestion_2014-01-02_1_config_019_doclayout_article_001.tiff
                -> 19
        """
        stem = Path(
            image_name
        ).stem

        match = CONFIG_ID_PATTERN.search(
            stem
        )

        if match is None:
            raise ValueError(
                "Could not extract enhancement "
                "config_id from filename: "
                f"{image_name}. Expected a filename "
                "containing '_config_<number>'."
            )

        return int(
            match.group(1)
        )

    @staticmethod
    def _extract_image_stem(
        image_name: str,
    ) -> str:
        """Extract canonical image_stem from filename.

        Examples:

            gestion_2014-01-02_1_config_019.tiff
                -> gestion_2014-01-02_1

            gestion_2014-01-02_1_config_019_doclayout_article_001.tiff
                -> gestion_2014-01-02_1
        """
        stem = Path(
            image_name
        ).stem

        match = re.match(
            r"^(?P<image_stem>.+?)_config_\d+(?:_|$)",
            stem,
        )

        if match is None:
            raise ValueError(
                "Could not extract image_stem from "
                f"filename: {image_name}."
            )

        return match.group(
            "image_stem"
        )

    def save_configurations(
        self,
        output_dir: Path,
    ) -> None:
        """Save the seven binarization methods to a text file."""
        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        config_file = (
            output_dir
            / CONFIGURATION_FILENAME
        )

        with config_file.open(
            "w",
            encoding="utf-8",
        ) as file:
            file.write(
                "BINARIZATION CONFIGURATIONS\n"
            )
            file.write(
                "===========================\n\n"
            )
            file.write(
                f"Total configurations: "
                f"{len(self.configs)}\n\n"
            )

            for idx, config in enumerate(
                self.configs
            ):
                file.write(
                    f"binarization_{idx:03d}\n"
                )
                file.write(
                    f"  binarization: "
                    f"{config['binarization']}\n\n"
                )

        self.logger.info(
            "Saved %d binarization configurations "
            "to %s",
            len(self.configs),
            config_file,
        )

    def _find_layout_record(
        self,
        relative_path: str,
        image_name: str,
    ) -> Optional[
        Tuple[str, Optional[str]]
    ]:
        """Find a cropped image in the layout Parquet.

        Matching is attempted using:

        1. The full relative input path.
        2. The input filename.
        3. The crop filename basename.

        Returns:
            Tuple of (image_stem, detector), or None
            if the image is not found in the layout metadata.
        """
        if self.layout_metadata is None:
            return None

        normalized_relative = (
            self._normalize_path(
                relative_path
            )
        )

        normalized_name = (
            self._normalize_path(
                image_name
            )
        )

        image_basename = Path(
            image_name
        ).name

        matches = self.layout_metadata[
            (
                self.layout_metadata[
                    "crop_file"
                ]
                == normalized_relative
            )
            |
            (
                self.layout_metadata[
                    "crop_file"
                ]
                == normalized_name
            )
            |
            (
                self.layout_metadata[
                    "crop_file"
                ]
                .map(Path)
                .map(
                    lambda path: path.name
                )
                == image_basename
            )
        ]

        if matches.empty:
            return None

        if len(matches) > 1:
            self.logger.warning(
                "Multiple layout records matched "
                "image %s. Using the first match.",
                image_name,
            )

        record = matches.iloc[0]

        image_stem = str(
            record["image_stem"]
        )

        detector = record[
            "detector"
        ]

        if pd.isna(detector):
            detector = None
        else:
            detector = str(
                detector
            )

        return (
            image_stem,
            detector,
        )

    def _resolve_image_metadata(
        self,
        image_name: str,
        relative_path: str,
    ) -> Tuple[
        str,
        Optional[str],
        int,
    ]:
        """Resolve image_stem, detector, and enhancement config_id.

        Metadata is extracted directly from the enhanced image filename.

        Full-page image pattern:
            <image_stem>_config_<config_id>.tiff

        Example:
            gestion_2014-01-02_1_config_000.tiff

        Resolves to:
            image_stem = "gestion_2014-01-02_1"
            config_id = 0
            detector = None

        Cropped image pattern:
            <image_stem>_config_<config_id>_<detector>_article_<id>.tiff

        Example:
            gestion_2014-01-02_1_config_1_histogram_article_002.tiff

        Resolves to:
            image_stem = "gestion_2014-01-02_1"
            config_id = 1
            detector = "histogram"

        The relative_path argument is retained for API compatibility
        and logging, but metadata resolution is based on image_name.
        """

        filename_stem = Path(
            image_name
        ).stem

        # ---------------------------------------------------------
        # Extract image_stem and config_id.
        #
        # This matches:
        #
        #   gestion_2014-01-02_1_config_000
        #   gestion_2014-01-02_1_config_1_histogram_article_002
        #
        # Group 1 -> image_stem
        # Group 2 -> config_id
        # Group 3 -> everything after config_id, if present
        # ---------------------------------------------------------

        match = re.match(
            r"^(?P<image_stem>.+?)"
            r"_config_(?P<config_id>\d+)"
            r"(?P<suffix>.*)$",
            filename_stem,
        )

        if match is None:
            raise ValueError(
                "Could not resolve image metadata from filename: "
                f"{image_name}. Expected a filename matching "
                "'<image_stem>_config_<config_id>[...]'."
            )

        image_stem = match.group(
            "image_stem"
        )

        config_id = int(
            match.group(
                "config_id"
            )
        )

        suffix = match.group(
            "suffix"
        )

        # ---------------------------------------------------------
        # Determine detector.
        #
        # Full-page:
        #
        #   gestion_2014-01-02_1_config_000
        #
        # suffix = ""
        # detector = None
        #
        # Cropped:
        #
        #   gestion_2014-01-02_1_config_1_histogram_article_002
        #
        # suffix = "_histogram_article_002"
        # detector = "histogram"
        # ---------------------------------------------------------

        detector = None

        if suffix:
            suffix_parts = [
                part
                for part in suffix.split("_")
                if part
            ]

            if suffix_parts:
                detector = suffix_parts[0]

        self.logger.debug(
            "Resolved metadata for %s: "
            "image_stem=%s, detector=%s, config_id=%d, "
            "relative_path=%s",
            image_name,
            image_stem,
            detector,
            config_id,
            relative_path,
        )

        return (
            image_stem,
            detector,
            config_id,
        )


    def process(
        self,
        image,
        image_name: str,
        relative_path: str,
        output_root: Path,
    ) -> Tuple[
        List[Dict],
        List[Dict],
        bool,
    ]:
        """Apply all seven binarization methods to one image.

        Returns:
            outputs:
                Successfully generated output images.

            records:
                Metadata records for all seven methods,
                including failed methods.

            all_successful:
                True only if all seven methods succeeded.
        """
        outputs: List[Dict] = []
        records: List[Dict] = []

        all_successful = True

        (
            image_stem,
            detector,
            config_id,
        ) = self._resolve_image_metadata(
            image_name=image_name,
            relative_path=relative_path,
        )

        self.logger.info(
            "[%s] image_stem=%s | "
            "detector=%s | "
            "enhancement_config_id=%d",
            image_name,
            image_stem,
            detector,
            config_id,
        )

        for binarization_idx, config in enumerate(
            self.configs
        ):
            binarization_name = (
                config["binarization"]
            )

            start = time.perf_counter()

            self.logger.info(
                "[%s] Binarization %d/%d | "
                "enhancement_config_id=%d | "
                "binarization=%s",
                image_name,
                binarization_idx + 1,
                len(self.configs),
                config_id,
                binarization_name,
            )

            output_filename = (
                f"{Path(image_name).stem}"
                f"_bin_"
                f"{binarization_idx:03d}.tiff"
            )

            output_relative = str(
                Path(relative_path).parent
                / output_filename
            )

            try:
                img = image.copy()

                method = getattr(
                    Binarization,
                    binarization_name,
                )

                img = method(img)

                elapsed = (
                    time.perf_counter()
                    - start
                )

                outputs.append(
                    {
                        "image": img,
                        "filename": (
                            output_filename
                        ),
                        "relative_parent": (
                            Path(
                                relative_path
                            ).parent
                        ),
                    }
                )

                records.append(
                    {
                        "image_stem": image_stem,
                        "detector": detector,
                        "config_id": config_id,
                        "binarization": (
                            binarization_name
                        ),
                        "binarize_file": (
                            output_relative
                        ),
                        "elapsed_s": round(
                            elapsed,
                            3,
                        ),
                        "status": "success",
                    }
                )

                self.logger.info(
                    "[%s] Binarization %d/%d "
                    "finished in %.3fs",
                    image_name,
                    binarization_idx + 1,
                    len(self.configs),
                    elapsed,
                )

            except Exception as exc:
                all_successful = False

                elapsed = (
                    time.perf_counter()
                    - start
                )

                self.logger.exception(
                    "[%s] FAILED binarization %d/%d "
                    "(%s): %s",
                    image_name,
                    binarization_idx + 1,
                    len(self.configs),
                    binarization_name,
                    exc,
                )

                records.append(
                    {
                        "image_stem": image_stem,
                        "detector": detector,
                        "config_id": config_id,
                        "binarization": (
                            binarization_name
                        ),
                        "binarize_file": None,
                        "elapsed_s": round(
                            elapsed,
                            3,
                        ),
                        "status": (
                            f"failed: {exc}"
                        ),
                    }
                )

        return (
            outputs,
            records,
            all_successful,
        )

    def _write_outputs(
        self,
        outputs: List[Dict],
        output_root: Path,
    ) -> bool:
        """Write all successfully processed images to disk."""
        for result in outputs:
            output_dir = (
                output_root
                / result["relative_parent"]
            )

            output_dir.mkdir(
                parents=True,
                exist_ok=True,
            )

            output_path = (
                output_dir
                / result["filename"]
            )

            success = cv2.imwrite(
                str(output_path),
                result["image"],
            )

            if not success:
                self.logger.error(
                    "Failed to write output image: %s",
                    output_path,
                )
                return False

        return True

    def _save_global_metadata(
        self,
        records: List[Dict],
        output_root: Path,
    ) -> None:
        """Save all metadata to one global Parquet.

        The Parquet is always written to:

            output_root/binarization.parquet

        Existing records are loaded and combined with the new
        records.

        Duplicate records are removed based on:

            image_stem
            detector
            config_id
            binarization
            binarize_file
        """
        parquet_path = (
            output_root
            / METADATA_FILENAME
        )

        if not records:
            self.logger.warning(
                "No metadata records to save. "
                "Parquet will not be created."
            )
            return

        df_new = pd.DataFrame(
            records
        )

        if parquet_path.exists():
            self.logger.info(
                "Loading existing metadata from %s",
                parquet_path,
            )

            df_existing = pd.read_parquet(
                parquet_path
            )

            df_combined = pd.concat(
                [
                    df_existing,
                    df_new,
                ],
                ignore_index=True,
            )

        else:
            df_combined = df_new

        unique_columns = [
            "image_stem",
            "detector",
            "config_id",
            "binarization",
            "binarize_file",
        ]

        df_combined = (
            df_combined
            .drop_duplicates(
                subset=unique_columns,
                keep="last",
            )
            .reset_index(drop=True)
        )

        # Explicitly enforce the expected column order.
        columns = [
            "image_stem",
            "detector",
            "config_id",
            "binarization",
            "binarize_file",
            "elapsed_s",
            "status",
        ]

        df_combined = df_combined[
            columns
        ]

        df_combined.to_parquet(
            parquet_path,
            index=False,
        )

        self.logger.info(
            "Saved global binarization metadata: "
            "%d records -> %s",
            len(df_combined),
            parquet_path,
        )

    def _load_checkpoint(
        self,
        checkpoint_path: Path,
    ) -> Set[str]:
        """Load successfully processed input paths."""
        if not checkpoint_path.exists():
            return set()

        with checkpoint_path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return {
                line.strip()
                for line in file
                if line.strip()
            }

    def _save_checkpoint(
        self,
        checkpoint_path: Path,
        relative_path: str,
    ) -> None:
        """Mark an input image as successfully processed."""
        checkpoint_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        with checkpoint_path.open(
            "a",
            encoding="utf-8",
        ) as file:
            file.write(
                relative_path
                + "\n"
            )

    def run(
        self,
        input_root: Path,
        output_root: Path,
        resume: bool = True,
    ) -> int:
        """Process all enhanced images.

        A single global Parquet is written at:

            output_root/binarization.parquet

        All full-page and cropped images are combined
        in that single file.
        """
        if not input_root.exists():
            raise FileNotFoundError(
                f"Input directory does not exist: "
                f"{input_root}"
            )

        if not input_root.is_dir():
            raise NotADirectoryError(
                f"Input path is not a directory: "
                f"{input_root}"
            )

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

        self.save_configurations(
            output_root
        )

        image_files = sorted(
            file_path
            for file_path in input_root.rglob("*")
            if (
                file_path.is_file()
                and file_path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )

        total = len(
            image_files
        )

        processed = 0
        skipped = 0
        failed = 0

        # All metadata records are accumulated here.
        # The Parquet is written once at the end.
        all_records: List[Dict] = []

        self.logger.info(
            "Found %d image(s) under %s",
            total,
            input_root,
        )

        for image_path in image_files:
            relative_path = str(
                image_path.relative_to(
                    input_root
                )
            )

            if (
                resume
                and relative_path in done
            ):
                self.logger.info(
                    "Skipping already processed "
                    "image: %s",
                    relative_path,
                )

                skipped += 1
                continue

            self.logger.info(
                "Processing image: %s",
                relative_path,
            )

            image = cv2.imread(
                str(image_path),
                cv2.IMREAD_UNCHANGED,
            )

            if image is None:
                self.logger.error(
                    "Could not read %s. "
                    "Skipping image.",
                    image_path,
                )

                failed += 1
                continue

            try:
                (
                    outputs,
                    records,
                    processing_successful,
                ) = self.process(
                    image=image,
                    image_name=image_path.name,
                    relative_path=relative_path,
                    output_root=output_root,
                )

            except Exception as exc:
                self.logger.exception(
                    "Failed to process %s: %s",
                    relative_path,
                    exc,
                )

                failed += 1
                continue

            if not processing_successful:
                self.logger.error(
                    "One or more binarization methods "
                    "failed for %s. "
                    "The image will not be checkpointed.",
                    relative_path,
                )

                # Keep failed records so they are represented
                # in the global Parquet.
                all_records.extend(
                    records
                )

                failed += 1
                continue

            outputs_written = (
                self._write_outputs(
                    outputs=outputs,
                    output_root=output_root,
                )
            )

            if not outputs_written:
                self.logger.error(
                    "Failed to write one or more "
                    "outputs for %s. "
                    "The image will not be checkpointed.",
                    relative_path,
                )

                # Keep metadata records even if writing failed.
                all_records.extend(
                    records
                )

                failed += 1
                continue

            # Add all seven successful metadata records
            # to the global metadata collection.
            all_records.extend(
                records
            )

            # Only checkpoint the input after all seven
            # binarization outputs were successfully generated
            # and written to disk.
            self._save_checkpoint(
                checkpoint_path,
                relative_path,
            )

            processed += 1

        # Save ONE global Parquet for the entire run.
        #
        # Location:
        #
        #     <output_root>/binarization.parquet
        #
        self._save_global_metadata(
            records=all_records,
            output_root=output_root,
        )

        self.logger.info(
            "Binarization complete. "
            "Processed: %d, Skipped: %d, "
            "Failed: %d, Total: %d",
            processed,
            skipped,
            failed,
            total,
        )

        return processed
