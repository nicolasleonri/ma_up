"""
Layout analysis pipeline with recursive year discovery, newspaper detector
routing, position-aware Trome config filtering, shard-safe checkpoints/Parquets,
and CPU worker parallelism.
"""

from concurrent.futures import ProcessPoolExecutor, as_completed
import json
import logging
from pathlib import Path
import time
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

from .steps import (
    Article,
    DocLayoutYOLODetector,
    HistogramColumnDetector,
    LayoutParserDetector,
    LayoutRegion,
    PPDocLayoutDetector,
    SuryaLayoutDetector,
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

GPU_DETECTORS = {"layoutparser", "doclayout", "ppdoclayout", "surya"}

# Winning layout detectors per newspaper (empty = full-page / no cropping)
NEWSPAPER_DETECTORS: Dict[str, List[str]] = {
    "correo": ["histogram"],
    "elcomercio": [],
    "gestion": [],
    "ojo": ["ppdoclayout"],
    "peru21": ["ppdoclayout"],
    "publimetro": ["histogram"],
    "trome": ["doclayout", "ppdoclayout"],
}

# Trome position-aware config_id -> detector routing:
#   config 7  (last page)      -> doclayout
#   config 34 (interior pages) -> ppdoclayout
TROME_CONFIG_DETECTOR: Dict[int, str] = {
    7: "doclayout",
    34: "ppdoclayout",
}

CHECKPOINT_FILENAME = ".layout_checkpoint.txt"


def _parse_image_stem(stem: str) -> Dict[str, str]:
    parts = stem.rsplit("_", 1)
    if len(parts) == 2:
        left, page = parts
        left_parts = left.rsplit("_", 1)
        if len(left_parts) == 2:
            newspaper, date = left_parts
            return {"newspaper": newspaper, "date": date, "page": page}
    return {"newspaper": stem, "date": "", "page": ""}


def _run_histogram_worker(
    variant_path_str: str,
    image_stem: str,
    config_id: int,
    output_dir_str: str,
    grid_rows: int,
    grid_cols: int,
    score_threshold: float,
    col_overlap_threshold: float,
    vertical_gap_ratio: float,
) -> tuple[List[Dict], Dict]:
    """Top-level picklable CPU worker for HistogramColumnDetector."""
    cv2.setNumThreads(1)
    variant_path = Path(variant_path_str)
    output_dir = Path(output_dir_str)
    detector_name = "histogram"
    start = time.time()

    image = cv2.imread(str(variant_path))
    if image is None:
        return [], {}

    h, w = image.shape[:2]
    try:
        detector = HistogramColumnDetector(score_threshold=score_threshold)
        regions = detector.detect(image)
        for region in regions:
            assign_grid_index(region, h, w, grid_rows, grid_cols)

        articles = group_regions_into_articles(
            regions,
            image_height=h,
            col_overlap_threshold=col_overlap_threshold,
            vertical_gap_ratio=vertical_gap_ratio,
        )

        crops_dir = output_dir / image_stem / f"config_{config_id}" / detector_name
        crops_dir.mkdir(parents=True, exist_ok=True)
        elapsed = time.time() - start

        rows: List[Dict] = []
        for article_idx, article in enumerate(articles, start=1):
            x1, y1 = max(0, article.x1), max(0, article.y1)
            x2, y2 = min(w, article.x2), min(h, article.y2)
            crop = image[y1:y2, x1:x2]
            filename = (
                f"{image_stem}_config_{config_id}_{detector_name}"
                f"_article_{article_idx:03d}.tiff"
            )
            cv2.imwrite(str(crops_dir / filename), crop)
            rows.append(
                {
                    "image_stem": image_stem,
                    "config_id": config_id,
                    "detector": detector_name,
                    "crop_file": filename,
                    "elapsed_s": round(elapsed, 3),
                    "status": "success",
                    "error": None,
                }
            )

        summary = {
            "image_stem": image_stem,
            "config_id": config_id,
            "detector": detector_name,
            "elapsed_s": round(elapsed, 3),
        }
        return rows, summary
    except Exception as exc:
        elapsed = time.time() - start
        return [
            {
                "image_stem": image_stem,
                "config_id": config_id,
                "detector": detector_name,
                "crop_file": None,
                "elapsed_s": round(elapsed, 3),
                "status": "failed",
                "error": str(exc),
            }
        ], {}


class LayoutAnalysisPipeline:
    def __init__(
        self,
        logger: logging.Logger,
        grid_rows: int = 3,
        grid_cols: int = 3,
        score_threshold: float = 0.5,
        col_overlap_threshold: float = 0.5,
        vertical_gap_ratio: float = 0.04,
        detectors: Optional[List[str]] = None,
        workers: int = 1,
        newspaper: Optional[str] = None,
    ):
        self.logger = logger
        self.grid_rows = grid_rows
        self.grid_cols = grid_cols
        self.score_threshold = score_threshold
        self.col_overlap_threshold = col_overlap_threshold
        self.vertical_gap_ratio = vertical_gap_ratio
        self.detector_names = (
            detectors if detectors is not None else list(DETECTORS.keys())
        )
        self.workers = max(1, workers)
        self.newspaper = newspaper.lower() if newspaper else None
        self._loaded_detectors: Dict[str, Any] = {}

    def _detectors_for_variant(
        self,
        config_id: int,
        candidate_detectors: List[str],
    ) -> List[str]:
        """Resolve which active detector(s) apply to a given (newspaper, config_id)."""
        if self.newspaper == "trome" and config_id in TROME_CONFIG_DETECTOR:
            target = TROME_CONFIG_DETECTOR[config_id]
            return [d for d in candidate_detectors if d == target]
        return candidate_detectors

    def _get_detector(self, name: str):
        if name not in self._loaded_detectors:
            self.logger.info("Loading detector: %s", name)
            self._loaded_detectors[name] = DETECTORS[name](
                score_threshold=self.score_threshold
            )
        return self._loaded_detectors[name]

    def _load_checkpoint(self, checkpoint_path: Path) -> set:
        if not checkpoint_path.exists():
            return set()
        with open(checkpoint_path, "r") as f:
            return {line.strip() for line in f if line.strip()}

    def _save_checkpoint(self, checkpoint_path: Path, key: str) -> None:
        with open(checkpoint_path, "a") as f:
            f.write(key + "\n")

    def process_variant(
        self,
        variant_path: Path,
        image_stem: str,
        config_id: int,
        output_dir: Path,
        detectors_to_run: List[str],
    ) -> tuple[List[Dict], List[Dict]]:
        """Run one or more detectors on a single preprocessed variant."""
        if not detectors_to_run:
            return [], []

        image = cv2.imread(str(variant_path))
        if image is None:
            self.logger.warning("Could not read variant: %s", variant_path)
            return [], []

        h, w = image.shape[:2]
        rows: List[Dict] = []
        results: List[Dict] = []

        for detector_name in detectors_to_run:
            start = time.time()
            try:
                detector = self._get_detector(detector_name)
                regions = detector.detect(image)

                for region in regions:
                    assign_grid_index(
                        region, h, w, self.grid_rows, self.grid_cols
                    )

                articles = group_regions_into_articles(
                    regions,
                    image_height=h,
                    col_overlap_threshold=self.col_overlap_threshold,
                    vertical_gap_ratio=self.vertical_gap_ratio,
                )

                crops_dir = (
                    output_dir
                    / image_stem
                    / f"config_{config_id}"
                    / detector_name
                )
                crops_dir.mkdir(parents=True, exist_ok=True)
                elapsed = time.time() - start

                for article_idx, article in enumerate(articles, start=1):
                    crop_path = self._save_article_crop(
                        image,
                        article,
                        crops_dir,
                        image_stem,
                        config_id,
                        detector_name,
                        article_idx,
                    )
                    rows.append(
                        {
                            "image_stem": image_stem,
                            "config_id": config_id,
                            "detector": detector_name,
                            "crop_file": crop_path.name,
                            "elapsed_s": round(elapsed, 3),
                            "status": "success",
                            "error": None,
                        }
                    )

                results.append(
                    {
                        "image_stem": image_stem,
                        "config_id": config_id,
                        "detector": detector_name,
                        "elapsed_s": round(elapsed, 3),
                    }
                )
            except Exception as exc:
                elapsed = time.time() - start
                self.logger.exception(
                    "[%s] FAILED config_%s detector=%s: %s",
                    image_stem,
                    config_id,
                    detector_name,
                    exc,
                )
                rows.append(
                    {
                        "image_stem": image_stem,
                        "config_id": config_id,
                        "detector": detector_name,
                        "crop_file": None,
                        "elapsed_s": round(elapsed, 3),
                        "status": "failed",
                        "error": str(exc),
                    }
                )

        return rows, results

    def run(
        self,
        preprocessed_dir: Path,
        output_dir: Path,
        resume: bool = True,
        sub_dirs: Optional[List[str]] = None,
        newspaper: Optional[str] = None,
        **kwargs: Any,
    ) -> int:
        if newspaper:
            self.newspaper = newspaper.lower()

        if not self.detector_names:
            self.logger.info(
                "No detectors configured for %s (full-page mode). Skipping layout detection.",
                preprocessed_dir,
            )
            return 0

        output_dir.mkdir(parents=True, exist_ok=True)

        det_tag = "_".join(sorted(self.detector_names))
        sub_tag = f"_{'_'.join(sub_dirs)}" if sub_dirs else ""
        checkpoint_path = (
            output_dir / f".layout_checkpoint{sub_tag}_{det_tag}.txt"
        )
        parquet_path = output_dir / f"layout_detection{sub_tag}_{det_tag}.parquet"

        done = self._load_checkpoint(checkpoint_path) if resume else set()

        scan_roots = (
            [
                preprocessed_dir / sd
                for sd in sub_dirs
                if (preprocessed_dir / sd).exists()
            ]
            if sub_dirs
            else [preprocessed_dir]
        )

        # Recursive discovery across year subdirectories (2013..2026)
        variant_files = sorted(
            f
            for root in scan_roots
            for f in root.rglob("*_config_*.tiff")
            if f.is_file()
        )

        if not variant_files:
            self.logger.warning(
                "No preprocessed variants found in %s", scan_roots
            )
            return 0

        # Group variants by image_stem
        stem_to_variants: Dict[str, List[Path]] = {}
        for vf in variant_files:
            stem = vf.stem.rsplit("_config_", 1)[0]
            stem_to_variants.setdefault(stem, []).append(vf)

        pending_stems = [
            stem for stem in sorted(stem_to_variants) if not (resume and stem in done)
        ]
        skipped = len(stem_to_variants) - len(pending_stems)

        self.logger.info(
            "Found %d image(s) (skipped=%d, pending=%d) | newspaper=%s | detectors=%s | workers=%d",
            len(stem_to_variants),
            skipped,
            len(pending_stems),
            self.newspaper,
            self.detector_names,
            self.workers,
        )

        if not pending_stems:
            return 0

        gpu_dets = [d for d in self.detector_names if d in GPU_DETECTORS]
        cpu_dets = [d for d in self.detector_names if d == "histogram"]

        processed = 0
        row_buffer: List[Dict] = []

        # 1. If pure-CPU histogram detector is requested and workers > 1, run in ProcessPoolExecutor
        histogram_results_by_stem: Dict[str, tuple[List[Dict], List[Dict]]] = {}
        if cpu_dets and self.workers > 1:
            tasks = []
            for stem in pending_stems:
                for vf in stem_to_variants[stem]:
                    cid = int(self._extract_config_id(vf))
                    if self._detectors_for_variant(cid, cpu_dets):
                        tasks.append((stem, vf, cid))

            with ProcessPoolExecutor(max_workers=self.workers) as pool:
                future_map = {
                    pool.submit(
                        _run_histogram_worker,
                        str(vf),
                        stem,
                        cid,
                        str(output_dir),
                        self.grid_rows,
                        self.grid_cols,
                        self.score_threshold,
                        self.col_overlap_threshold,
                        self.vertical_gap_ratio,
                    ): stem
                    for stem, vf, cid in tasks
                }
                for fut in as_completed(future_map):
                    stem = future_map[fut]
                    r_list, res_obj = fut.result()
                    s_rows, s_res = histogram_results_by_stem.setdefault(
                        stem, ([], [])
                    )
                    s_rows.extend(r_list)
                    if res_obj:
                        s_res.append(res_obj)

        # 2. Run GPU detectors (and workers==1 histogram) in-process with preloaded models
        for stem in pending_stems:
            stem_rows, stem_results = histogram_results_by_stem.get(stem, ([], []))

            in_process_dets = (
                gpu_dets if (cpu_dets and self.workers > 1) else self.detector_names
            )

            if in_process_dets:
                for vf in stem_to_variants[stem]:
                    cid = int(self._extract_config_id(vf))
                    active_dets = self._detectors_for_variant(cid, in_process_dets)
                    if not active_dets:
                        continue
                    r_list, res_list = self.process_variant(
                        vf, stem, cid, output_dir, active_dets
                    )
                    stem_rows.extend(r_list)
                    stem_results.extend(res_list)

            if stem_rows:
                row_buffer.extend(stem_rows)
            if stem_results:
                self._save_index(stem_results, output_dir / stem)

            self._save_checkpoint(checkpoint_path, stem)
            processed += 1

            if processed % 50 == 0:
                if row_buffer:
                    self._append_to_parquet(row_buffer, parquet_path)
                    row_buffer.clear()
                self.logger.info(
                    "Progress: %d/%d pending images completed",
                    processed,
                    len(pending_stems),
                )

        if row_buffer:
            self._append_to_parquet(row_buffer, parquet_path)

        self.logger.info(
            "Layout analysis complete. Processed: %d, Skipped: %d, Total: %d",
            processed,
            skipped,
            len(stem_to_variants),
        )
        return processed

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
        import pandas as pd

        new_df = pd.DataFrame(rows)
        key_cols = ["image_stem", "config_id", "detector", "crop_file"]

        if parquet_path.exists():
            existing_df = pd.read_parquet(parquet_path)
            combined = pd.concat([existing_df, new_df], ignore_index=True)
            combined = combined.drop_duplicates(subset=key_cols, keep="last")
        else:
            combined = new_df

        combined = combined.sort_values(
            ["image_stem", "config_id", "detector"]
        ).reset_index(drop=True)
        combined.to_parquet(parquet_path, index=False)

    def _save_index(self, results: List[Dict], image_output_dir: Path) -> None:
        image_output_dir.mkdir(parents=True, exist_ok=True)
        index_path = image_output_dir / "layout_index.json"
        with open(index_path, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)