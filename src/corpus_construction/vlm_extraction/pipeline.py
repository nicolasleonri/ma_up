"""
VLM extraction pipeline.

Reads successful binarized TIFFs listed in the binarization Parquet
and runs one or more local VLMs over each image.

Architecture:

    binarized image
        ↓
    local in-process VLM via vLLM
        ↓
    DSPy structured extraction
        ↓
    title / subheadline / author / body
        ↓
    global Parquet

No VLM server is required.

Each VLM receives every successful binarization variant as a separate
input. This allows evaluation across:

    config_id
        ×
    detector
        ×
    binarization
        ×
    VLM

Output schema
-------------

image_stem
config_id
detector
binarization
binarize_file
vlm
title
subheadline
author
body
raw_text
elapsed_s
status
error
"""

import logging
from pathlib import Path
from typing import Any, Dict, List

import cv2
import pandas as pd

from .steps import (
    ExtractionResult,
    VLM_EXTRACTORS,
)


DEFAULT_PARQUET = (
    "data/corpus_construction/"
    "vlm_extraction/results.parquet"
)


# ----------------------------------------------------------------------
# Unique extraction key
# ----------------------------------------------------------------------

_KEY_COLS = [
    "image_stem",
    "config_id",
    "detector",
    "binarization",
    "binarize_file",
    "vlm",
]


class VLMExtractionPipeline:
    """
    Run local in-process VLM extraction over successful binarization
    outputs.

    The binarization Parquet is the metadata contract between the
    binarization and VLM stages.

    Each successful row represents one unique image input for VLM
    extraction.
    """

    def __init__(
        self,
        logger: logging.Logger,
        vlms: List[str] = None,
        binarization_parquet: str = "",
        binarized_dir: str = "",
        parquet_path: str = DEFAULT_PARQUET,
        max_new_tokens: int = 2048,
        batch_size: int = 16,
        skip_failed_extractions: bool = True,
        vlm_engine_kwargs: Dict[
            str,
            Dict[str, Any],
        ] = None,
    ):
        self.logger = logger

        self.vlm_names = (
            vlms
            or list(VLM_EXTRACTORS.keys())
        )

        self.binarization_parquet = Path(
            binarization_parquet
        )

        self.binarized_dir = Path(
            binarized_dir
        )

        self.parquet_path = Path(
            parquet_path
        )

        self.max_new_tokens = (
            max_new_tokens
        )

        self.batch_size = (
            batch_size
        )

        self.skip_failed_extractions = (
            skip_failed_extractions
        )

        self.vlm_engine_kwargs = (
            vlm_engine_kwargs
            or {}
        )

        self._loaded_vlms = {}

    # ------------------------------------------------------------------
    # VLM cache
    # ------------------------------------------------------------------

    def _get_vlm(
        self,
        name: str,
    ):
        """
        Load a VLM lazily and cache it.

        VLM inference is performed locally in-process.
        No HTTP/server endpoint is used.
        """

        if name not in self._loaded_vlms:

            if name not in VLM_EXTRACTORS:
                raise ValueError(
                    f"Unknown VLM: {name!r}. "
                    f"Choose from "
                    f"{list(VLM_EXTRACTORS)}"
                )

            self.logger.info(
                "Loading local VLM extractor: %s",
                name,
            )

            extractor_cls = (
                VLM_EXTRACTORS[name]
            )

            self._loaded_vlms[name] = (
                extractor_cls(
                    max_new_tokens=(
                        self.max_new_tokens
                    ),
                    **(
                        self.vlm_engine_kwargs.get(
                            name,
                            {},
                        )
                    ),
                )
            )

        return self._loaded_vlms[name]

    # ------------------------------------------------------------------
    # Already-processed set
    # ------------------------------------------------------------------

    def _load_done_keys(
        self,
    ) -> set:
        """
        Return successfully completed extraction keys.

        Failed rows are excluded when
        skip_failed_extractions=True so they can be retried.
        """

        if not self.parquet_path.exists():
            return set()

        # --------------------------------------------------------------
        # Important:
        #
        # The output Parquet may have been created by an older version
        # of the pipeline. Do not assume that it contains all expected
        # columns. This avoids the ArrowInvalid error:
        #
        #   No match for FieldRef.Name(config_id)
        #
        # We inspect the schema first.
        # --------------------------------------------------------------

        try:
            available_columns = set(
                pd.read_parquet(
                    self.parquet_path,
                    engine="pyarrow",
                ).columns
            )
        except Exception as exc:
            self.logger.warning(
                "Could not inspect existing VLM Parquet %s: %s",
                self.parquet_path,
                exc,
            )
            return set()

        required_columns = set(
            _KEY_COLS + ["status"]
        )

        missing = (
            required_columns
            - available_columns
        )

        if missing:
            self.logger.warning(
                "Existing VLM Parquet is incompatible "
                "with the current schema. Missing columns: %s. "
                "Existing results will not be used for resume.",
                sorted(missing),
            )
            return set()

        df = pd.read_parquet(
            self.parquet_path,
            columns=(
                _KEY_COLS
                + ["status"]
            ),
        )

        if self.skip_failed_extractions:
            df = df[
                df["status"] == "success"
            ]

        return set(
            zip(
                *(
                    df[column]
                    for column in _KEY_COLS
                )
            )
        )

    # ------------------------------------------------------------------
    # Binarization input discovery
    # ------------------------------------------------------------------

    def _discover_inputs(
        self,
    ) -> List[Dict[str, Any]]:
        """
        Read the binarization Parquet and return all successful
        binarized images that exist on disk.
        """

        df = pd.read_parquet(
            self.binarization_parquet
        )

        required_columns = {
            "image_stem",
            "detector",
            "config_id",
            "binarization",
            "binarize_file",
            "status",
        }

        missing = (
            required_columns
            - set(df.columns)
        )

        if missing:
            raise ValueError(
                "Binarization Parquet is missing "
                f"required columns: {sorted(missing)}"
            )

        df = df[
            df["status"] == "success"
        ].copy()

        inputs = []

        for _, row in df.iterrows():

            binarize_file = str(
                row["binarize_file"]
            )

            image_path = (
                self.binarized_dir
                / binarize_file
            )

            if not image_path.exists():

                self.logger.warning(
                    "Binarized file missing: %s",
                    image_path,
                )

                continue

            detector = row["detector"]

            if pd.isna(detector):
                detector = None

            inputs.append(
                {
                    "image_path": image_path,
                    "image_stem": str(
                        row["image_stem"]
                    ),
                    "detector": detector,
                    "config_id": int(
                        row["config_id"]
                    ),
                    "binarization": str(
                        row["binarization"]
                    ),
                    "binarize_file": (
                        binarize_file
                    ),
                }
            )

        return inputs

    # ------------------------------------------------------------------
    # Core run
    # ------------------------------------------------------------------

    def run(self) -> int:
        """
        Run all requested local VLMs over all successful binarization
        outputs.

        Returns
        -------
        int
            Number of VLM extraction results written.
        """

        if not self.binarization_parquet.exists():

            self.logger.error(
                "Binarization Parquet not found: %s",
                self.binarization_parquet,
            )

            return 0

        inputs = (
            self._discover_inputs()
        )

        if not inputs:

            self.logger.warning(
                "No usable binarized images found."
            )

            return 0

        done_keys = (
            self._load_done_keys()
        )

        self.logger.info(
            "Binarized images available: %d",
            len(inputs),
        )

        self.logger.info(
            "Already completed extractions: %d",
            len(done_keys),
        )

        self.parquet_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        processed = 0

        # --------------------------------------------------------------
        # Process one VLM at a time.
        # --------------------------------------------------------------

        for vlm_name in self.vlm_names:

            pending = []

            for item in inputs:

                key = (
                    item["image_stem"],
                    item["config_id"],
                    item["detector"],
                    item["binarization"],
                    item["binarize_file"],
                    vlm_name,
                )

                if key not in done_keys:
                    pending.append(item)

            if not pending:

                self.logger.info(
                    "[%s] Nothing pending.",
                    vlm_name,
                )

                continue

            self.logger.info(
                "[%s] %d images pending.",
                vlm_name,
                len(pending),
            )

            extractor = self._get_vlm(
                vlm_name
            )

            # ----------------------------------------------------------
            # Process in batches.
            # ----------------------------------------------------------

            for chunk_start in range(
                0,
                len(pending),
                self.batch_size,
            ):

                chunk = pending[
                    chunk_start:
                    chunk_start
                    + self.batch_size
                ]

                images = []
                metadata_list = []

                for item in chunk:

                    image = cv2.imread(
                        str(
                            item["image_path"]
                        )
                    )

                    if image is None:

                        self.logger.warning(
                            "Could not read image: %s",
                            item["image_path"],
                        )

                        continue

                    metadata = {
                        "image_stem": (
                            item["image_stem"]
                        ),
                        "config_id": (
                            item["config_id"]
                        ),
                        "detector": (
                            item["detector"]
                        ),
                        "binarization": (
                            item["binarization"]
                        ),
                        "binarize_file": (
                            item["binarize_file"]
                        ),
                        "vlm": vlm_name,
                    }

                    images.append(image)

                    metadata_list.append(
                        metadata
                    )

                if not images:
                    continue

                self.logger.info(
                    "[%s] Batch %d-%d / %d",
                    vlm_name,
                    chunk_start,
                    (
                        chunk_start
                        + len(images)
                    ),
                    len(pending),
                )

                results: List[
                    ExtractionResult
                ] = extractor.extract_batch(
                    images,
                    metadata_list,
                )

                for result in results:

                    self.logger.info(
                        (
                            "  → status=%s | "
                            "title=%r | "
                            "subheadline=%r | "
                            "author=%r | "
                            "body_chars=%d | "
                            "%.3fs"
                        ),
                        result.status,
                        (
                            result.title[:60]
                            if result.title
                            else ""
                        ),
                        (
                            result.subheadline[:60]
                            if result.subheadline
                            else ""
                        ),
                        (
                            result.author[:60]
                            if result.author
                            else ""
                        ),
                        len(result.body),
                        result.elapsed_s,
                    )

                rows = [
                    result.to_dict()
                    for result in results
                ]

                self._append_to_parquet(
                    rows
                )

                processed += len(
                    results
                )

                # Add newly completed successful rows to the in-memory
                # resume set so duplicate work is avoided during the
                # same run.
                for result in results:

                    if result.status != "success":
                        continue

                    key = tuple(
                        result.metadata.get(
                            col
                        )
                        for col in _KEY_COLS
                    )

                    done_keys.add(key)

            # ----------------------------------------------------------
            # Free GPU memory before loading next VLM.
            # ----------------------------------------------------------

            self.logger.info(
                "[%s] Done. Unloading model.",
                vlm_name,
            )

            extractor.unload()

            del self._loaded_vlms[
                vlm_name
            ]

        self.logger.info(
            "VLM extraction complete. "
            "Extractions run: %d",
            processed,
        )

        return processed

    # ------------------------------------------------------------------
    # Parquet I/O
    # ------------------------------------------------------------------

    def _append_to_parquet(
        self,
        rows: List[Dict],
    ) -> None:
        """
        Append extraction results to the global Parquet.

        Existing rows with the same unique extraction key are replaced
        by the newest result.
        """

        if not rows:
            return

        new_df = pd.DataFrame(
            rows
        )

        if self.parquet_path.exists():

            try:
                existing = pd.read_parquet(
                    self.parquet_path
                )

                # If the existing file comes from an incompatible
                # previous schema, discard it rather than allowing
                # concat/drop_duplicates to silently produce corruption.
                if not set(_KEY_COLS).issubset(
                    existing.columns
                ):
                    self.logger.warning(
                        "Existing VLM Parquet has an incompatible "
                        "schema. Replacing it with the current schema."
                    )
                    existing = pd.DataFrame()

            except Exception as exc:

                self.logger.warning(
                    "Could not read existing VLM Parquet: %s. "
                    "Replacing it.",
                    exc,
                )

                existing = pd.DataFrame()

            combined = pd.concat(
                [
                    existing,
                    new_df,
                ],
                ignore_index=True,
            )

            combined = (
                combined.drop_duplicates(
                    subset=_KEY_COLS,
                    keep="last",
                )
            )

        else:

            combined = new_df

        combined["_detector_sort"] = (
            combined["detector"]
            .fillna("")
            .astype(str)
        )

        combined = (
            combined.sort_values(
                [
                    "image_stem",
                    "config_id",
                    "_detector_sort",
                    "binarization",
                    "vlm",
                ]
            )
            .drop(
                columns=[
                    "_detector_sort"
                ]
            )
            .reset_index(
                drop=True
            )
        )

        combined.to_parquet(
            self.parquet_path,
            index=False,
        )

        self.logger.info(
            "Saved %d new rows to %s "
            "(total: %d)",
            len(new_df),
            self.parquet_path,
            len(combined),
        )