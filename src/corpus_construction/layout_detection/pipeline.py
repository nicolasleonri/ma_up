"""
Layout analysis pipeline.

Output per article crop
-----------------------
Filename: {newspaper}_{date}_{page}_config_{N}_{detector}_article_{M}.tiff

Parquet schema (one row per article crop)
-----------------------------------------
newspaper          str     e.g. "elcomercio"
date               str     e.g. "2026-01-03"
page               str     e.g. "2"
image_stem         str     full original stem, e.g. "elcomercio_2026-01-03_2"
preprocessing_config int   config index from preprocessing step
detector           str     "layoutparser" | "doclayout" | "ppdoclayout" | "surya"
article_idx        int     1-based article number (top-to-bottom)
crop_file          str     filename of the saved crop
x1, y1, x2, y2    int     merged bounding box of the article
grid_row           int     row in composition grid
grid_col           int     col in composition grid
num_regions        int     number of detected regions merged into this article
elapsed_s          float   wall-clock seconds for this detector × config run
status             str     "ok" | "failed"
error              str     error message if status == "failed", else null
"""

import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional

import cv2
import numpy as np


from .steps import (
    LayoutParserDetector,
    DocLayoutYOLODetector,
    PPDocLayoutDetector,
    SuryaLayoutDetector,
    HistogramColumnDetector,
    LayoutRegion,
    Article,
    assign_grid_index,
    group_regions_into_articles,
)


DETECTORS = {
    "layoutparser": LayoutParserDetector,
    "doclayout": DocLayoutYOLODetector,
    "ppdoclayout": PPDocLayoutDetector,
    "surya": SuryaLayoutDetector,
    "histogram": HistogramColumnDetector,
}

CHECKPOINT_FILENAME = ".layout_checkpoint.txt"


def _parse_image_stem(stem: str) -> Dict[str, str]:
    """
    Extract newspaper / date / page from a stem like 'elcomercio_2026-01-03_2'.
    Falls back gracefully if the stem doesn't follow the convention.
    """
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        # parts[0] = "elcomercio_2026-01-03", parts[1] = "2"
        left, page = parts
        left_parts = left.rsplit("_", 1)
        if len(left_parts) == 2:
            newspaper, date = left_parts
            return {"newspaper": newspaper, "date": date, "page": page}
    return {"newspaper": stem, "date": "", "page": ""}


class LayoutAnalysisPipeline:
    """
    Runs LayoutParser and DocLayout-YOLO on every preprocessed variant of a
    newspaper image. For each variant it:

      1. Detects editorial regions (headlines, body text, captions).
      2. Groups regions into coherent articles (column + vertical proximity).
      3. Crops and saves each article as a TIFF with full provenance in the name.
      4. Appends one row per article crop to a shared Parquet results file.
      5. Writes a per-image JSON index for quick inspection.
    """

    def __init__(
        self,
        logger: logging.Logger,
        grid_rows: int = 3,
        grid_cols: int = 3,
        score_threshold: float = 0.5,
        col_overlap_threshold: float = 0.5,
        vertical_gap_ratio: float = 0.04,
        detectors: List[str] = None,
    ):
        self.logger = logger
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.score_threshold = score_threshold
        self.col_overlap_threshold = col_overlap_threshold
        self.vertical_gap_ratio = vertical_gap_ratio
        self.detector_names = detectors or list(DETECTORS.keys())
        self._loaded_detectors: Dict[str, Any] = {}

    # ------------------------------------------------------------------
    # Detector cache
    # ------------------------------------------------------------------

    def _get_detector(self, name: str):
        if name not in self._loaded_detectors:
            self.logger.info("Loading detector: %s", name)
            self._loaded_detectors[name] = DETECTORS[name](
                score_threshold=self.score_threshold
            )
        return self._loaded_detectors[name]

    # ------------------------------------------------------------------
    # Checkpoint
    # ------------------------------------------------------------------

    def _load_checkpoint(self, checkpoint_path: Path) -> set:
        if not checkpoint_path.exists():
            return set()
        with open(checkpoint_path, "r") as f:
            return {line.strip() for line in f if line.strip()}

    def _save_checkpoint(self, checkpoint_path: Path, stem: str) -> None:
        with open(checkpoint_path, "a") as f:
            f.write(stem + "\n")

    # ------------------------------------------------------------------
    # Core processing
    # ------------------------------------------------------------------

    def process(
        self,
        preprocessed_dir: Path,
        image_stem: str,
        output_dir: Path,
    ) -> List[Dict]:
        """
        Process all preprocessed variants of one original image.

        Returns a flat list of row dicts (one per article crop) ready to be
        appended to the Parquet results file.
        """
        variants = sorted(preprocessed_dir.glob(f"{image_stem}_config_*.tiff"))

        if not variants:
            self.logger.warning(
                "No preprocessed variants found for %s in %s",
                image_stem, preprocessed_dir,
            )
            return []

        meta = _parse_image_stem(image_stem)
        all_rows: List[Dict] = []
        all_results: List[Dict] = []  # for JSON index

        for variant_path in variants:
            config_id = int(self._extract_config_id(variant_path))
            image = cv2.imread(str(variant_path))

            if image is None:
                self.logger.warning("Could not read variant: %s", variant_path)
                continue

            h, w = image.shape[:2]

            for detector_name in self.detector_names:
                start = time.time()
                self.logger.info(
                    "[%s] config_%s | detector=%s",
                    image_stem, config_id, detector_name,
                )

                try:
                    detector = self._get_detector(detector_name)
                    regions = detector.detect(image)

                    for region in regions:
                        assign_grid_index(region, h, w, self.grid_rows, self.grid_cols)

                    articles = group_regions_into_articles(
                        regions,
                        image_height=h,
                        col_overlap_threshold=self.col_overlap_threshold,
                        vertical_gap_ratio=self.vertical_gap_ratio,
                    )

                    crops_dir = (
                        output_dir / image_stem
                        / f"config_{config_id}"
                        / detector_name
                    )
                    crops_dir.mkdir(parents=True, exist_ok=True)

                    elapsed = time.time() - start
                    article_records = []

                    for article_idx, article in enumerate(articles, start=1):
                        crop_path = self._save_article_crop(
                            image, article, crops_dir,
                            image_stem, config_id, detector_name, article_idx,
                        )

                        row = {
                            # Provenance
                            "newspaper": meta["newspaper"],
                            "date": meta["date"],
                            "page": meta["page"],
                            "image_stem": image_stem,
                            "preprocessing_config": config_id,
                            "detector": detector_name,
                            "article_idx": article_idx,
                            # Output
                            "crop_file": crop_path.name,
                            "crop_path": str(crop_path),
                            # Geometry
                            "x1": article.x1,
                            "y1": article.y1,
                            "x2": article.x2,
                            "y2": article.y2,
                            "grid_row": article.grid_row,
                            "grid_col": article.grid_col,
                            "num_regions": len(article.regions),
                            # Timing
                            "elapsed_s": round(elapsed, 3),
                            # Status
                            "status": "ok",
                            "error": None,
                        }
                        all_rows.append(row)

                        record = article.to_dict()
                        record["crop_file"] = crop_path.name
                        article_records.append(record)

                    all_results.append({
                        "image": image_stem,
                        "preprocessing_config": config_id,
                        "detector": detector_name,
                        "num_regions_detected": len(regions),
                        "num_articles": len(articles),
                        "elapsed_s": round(elapsed, 3),
                        "articles": article_records,
                    })

                    self.logger.info(
                        "[%s] config_%s | detector=%s → %d regions → %d articles in %.3fs",
                        image_stem, config_id, detector_name,
                        len(regions), len(articles), elapsed,
                    )

                except Exception as e:
                    elapsed = time.time() - start
                    self.logger.exception(
                        "[%s] FAILED config_%s detector=%s: %s",
                        image_stem, config_id, detector_name, e,
                    )
                    all_rows.append({
                        "newspaper": meta["newspaper"],
                        "date": meta["date"],
                        "page": meta["page"],
                        "image_stem": image_stem,
                        "preprocessing_config": config_id,
                        "detector": detector_name,
                        "article_idx": None,
                        "crop_file": None,
                        "crop_path": None,
                        "x1": None, "y1": None, "x2": None, "y2": None,
                        "grid_row": None, "grid_col": None,
                        "num_regions": None,
                        "elapsed_s": round(elapsed, 3),
                        "status": "failed",
                        "error": str(e),
                    })

        if all_results:
            self._save_index(all_results, output_dir / image_stem)

        return all_rows

    # ------------------------------------------------------------------
    # Run over a full directory
    # ------------------------------------------------------------------

    def run(
        self,
        preprocessed_dir: Path,
        output_dir: Path,
        resume: bool = True,
    ) -> int:
        """
        Run layout analysis on all preprocessed images in preprocessed_dir.
        Appends results to a Parquet file saved inside output_dir after each
        image. Supports resuming interrupted runs via a checkpoint file.

        Returns the number of original images processed.
        """
        output_dir.mkdir(parents=True, exist_ok=True)

        checkpoint_path = output_dir / CHECKPOINT_FILENAME
        parquet_path = output_dir / "layout_detection.parquet"

        done = self._load_checkpoint(checkpoint_path) if resume else set()

        stems = set()
        for f in preprocessed_dir.glob("*_config_*.tiff"):
            stem = f.stem.rsplit("_config_", 1)[0]
            stems.add(stem)

        if not stems:
            self.logger.warning(
                "No preprocessed variants found in %s", preprocessed_dir
            )
            return 0

        self.logger.info(
            "Found %d original image(s) to process with %s detector(s).",
            len(stems), self.detector_names,
        )

        processed = 0
        skipped = 0

        for stem in sorted(stems):
            if resume and stem in done:
                skipped += 1
                continue

            self.logger.info("Starting layout analysis: %s", stem)
            rows = self.process(preprocessed_dir, stem, output_dir)

            if rows:
                self._append_to_parquet(rows, parquet_path)

            self._save_checkpoint(checkpoint_path, stem)
            processed += 1

        self.logger.info(
            "Layout analysis complete. Processed: %d, Skipped: %d, Total: %d",
            processed, skipped, len(stems),
        )
        return processed

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _extract_config_id(self, path: Path) -> str:
        return path.stem.rsplit("_config_", 1)[-1]

    def _save_article_crop(
        self,
        image: np.ndarray,
        article: Article,
        crops_dir: Path,
        image_stem: str,
        config_id: int,
        detector_name: str,
        article_idx: int,
    ) -> Path:
        """
        Crop the merged article bounding box and save with full provenance name.
        e.g. elcomercio_2026-01-03_2_config_1_layoutparser_article_001.tiff
        """
        h, w = image.shape[:2]
        x1 = max(0, article.x1)
        y1 = max(0, article.y1)
        x2 = min(w, article.x2)
        y2 = min(h, article.y2)

        crop = image[y1:y2, x1:x2]
        filename = (
            f"{image_stem}"
            f"_config_{config_id}"
            f"_{detector_name}"
            f"_article_{article_idx:03d}"
            ".tiff"
        )
        out_path = crops_dir / filename
        cv2.imwrite(str(out_path), crop)
        return out_path

    def _append_to_parquet(self, rows: List[Dict], parquet_path: Path) -> None:
        """Append rows to the Parquet results file (upsert by key cols)."""
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError(
                "Saving results requires 'pandas' and 'pyarrow'. "
                "Install with: pip install pandas pyarrow"
            ) from exc

        new_df = pd.DataFrame(rows)

        key_cols = ["image_stem", "preprocessing_config", "detector", "article_idx"]

        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = new_df

        combined = combined.sort_values(
            ["newspaper", "date", "page", "preprocessing_config", "detector", "article_idx"]
        ).reset_index(drop=True)
        combined.to_parquet(parquet_path, index=False)

        self.logger.info(
            "Appended %d row(s) to %s (total rows: %d)",
            len(new_df), parquet_path, len(combined),
        )

    def _save_index(self, results: List[Dict], image_output_dir: Path) -> None:
        image_output_dir.mkdir(parents=True, exist_ok=True)
        index_path = image_output_dir / "layout_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        self.logger.info("Saved layout index: %s", index_path)