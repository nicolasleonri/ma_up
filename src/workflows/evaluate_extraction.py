"""
evaluate_extraction.py

Evaluate OCR/LLM extraction results against a gold-standard CSV.

Detection and text quality are measured separately:

    Detection metrics
        - TP / FP / FN
        - precision / recall / F1

    Text quality metrics
        - CER / WER for alignable matched article pairs only
        - arithmetic mean across matched pairs

Matching:
    score = 0.30 * fuzzy_title + 0.70 * fuzzy_body
    Hungarian one-to-one assignment
    score >= 70          -> TP
    20 <= score < 70     -> FP (alignable, so CER/WER are still measured)
    score < 20           -> not alignable
    unmatched prediction -> FP
    FN count = gold_articles - TP

Timing:
    One successful pipeline combination corresponds to one pipeline execution.
    A single LLM response may contain multiple articles, but its LLM time is
    counted once. Article-level rows repeat the combo execution time only for
    traceability; time must be aggregated from page metrics, not article rows.

Outputs:
    <output stem>_results.csv      article-level TP/FP/FN rows
    <output stem>_page_metrics.csv page/config-level detection + quality + timing

Example:
    --output-csv results/correo_results.csv
    -> results/correo_results.csv
    -> results/correo_page_metrics.csv
"""

import argparse
import csv
import json
import math
import random
import re
import unicodedata
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm
import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment


# =========================================================
# Configuration
# =========================================================

COMBO_COLS = [
    "image_stem",
    "config_id",
    "layout_mode",
    "detector",
    "binarization",
    "ocr_extractor",
    "llm_extractor",
]

MATCH_THRESHOLD = 70.0
MIN_MATCH_SCORE = 20.0

TITLE_WEIGHT = 0.30
BODY_WEIGHT = 0.70
AUDIT_EVERY = 50

DETECTOR_NONE = "__none__"

ARTICLE_FIELDS = [
    "image_stem",
    "config_id",
    "layout_mode",
    "detector",
    "binarization",
    "ocr_extractor",
    "llm_extractor",
    "result_type",
    "pred_title",
    "pred_body",
    "gold_title",
    "gold_body",
    "matching_score",
    "title_fuzzy_score",
    "body_fuzzy_score",
    "title_cer",
    "title_wer",
    "body_cer",
    "body_wer",
    "body_len_delta_chars",
    "total_pipeline_seconds",
]

PAGE_FIELDS = [
    "image_stem",
    "config_id",
    "layout_mode",
    "detector",
    "binarization",
    "ocr_extractor",
    "llm_extractor",
    "gold_articles",
    "predicted_articles",
    "tp",
    "fp",
    "fn",
    "detection_precision",
    "detection_recall",
    "detection_f1",
    "matched_pairs_for_error_mean",
    "matched_title_cer_mean",
    "matched_title_wer_mean",
    "matched_body_cer_mean",
    "matched_body_wer_mean",
    "total_pipeline_seconds",
    "parse_failure_rows",
]


# =========================================================
# Generic helpers
# =========================================================

def is_missing(value: Any) -> bool:
    """Safely test whether a scalar pandas value is missing."""
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict, set, np.ndarray)):
        return False
    try:
        result = pd.isna(value)
        return bool(result)
    except (TypeError, ValueError):
        return False


def safe_text(value: Any) -> str:
    """
    Convert a pandas value to a safe string.

    NaN / None become "".
    """
    if is_missing(value):
        return ""
    return str(value)


def normalize_text(text: Any) -> str:
    """
    Normalize text for fuzzy matching.

    - Unicode NFKD normalization
    - remove combining accents
    - lowercase
    - punctuation -> spaces
    - collapse whitespace
    """
    text = safe_text(text)

    if not text:
        return ""

    text = unicodedata.normalize("NFKD", text)
    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.lower()
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def csv_value(value: Any) -> Any:
    """Convert NaN-like values to None for clean CSV output."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value


def preview(text: Any, max_chars: int = 700) -> str:
    """Compact one-line preview for console audit output."""
    text = safe_text(text).replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) <= max_chars:
        return text
    return text[:max_chars] + " ..."


def combo_meta(
    image_stem: Any,
    config_id: Any,
    layout_mode: Any,
    detector: Any,
    binarization: Any,
    ocr_extractor: Any,
    llm_extractor: Any,
) -> Dict[str, Any]:
    """Build user-facing combo metadata."""
    return {
        "image_stem": image_stem,
        "config_id": config_id,
        "layout_mode": layout_mode,
        "detector": (
            None
            if safe_text(detector) == DETECTOR_NONE
            else detector
        ),
        "binarization": binarization,
        "ocr_extractor": ocr_extractor,
        "llm_extractor": llm_extractor,
    }


# =========================================================
# JSON article parsing
# =========================================================

def strip_markdown_fence(text: str) -> str:
    """Remove a surrounding Markdown code fence, if present."""
    text = text.strip()
    text = re.sub(
        r"^\s*```(?:json|javascript)?\s*",
        "",
        text,
        flags=re.IGNORECASE,
    )
    text = re.sub(r"\s*```\s*$", "", text)
    return text.strip()


def repair_json_control_chars(text: str) -> str:
    """
    Escape literal control characters that occur inside JSON strings.

    LLM output occasionally contains literal newlines/tabs/etc. inside quoted
    JSON string values. json.loads() rejects those even though the intended
    value is clear.
    """
    out: List[str] = []
    in_string = False
    escaped = False

    for ch in text:
        code = ord(ch)

        if in_string:
            if escaped:
                out.append(ch)
                escaped = False
                continue

            if ch == "\\":
                out.append(ch)
                escaped = True
                continue

            if ch == '"':
                out.append(ch)
                in_string = False
                continue

            if code < 32:
                if ch == "\n":
                    out.append("\\n")
                elif ch == "\r":
                    out.append("\\r")
                elif ch == "\t":
                    out.append("\\t")
                else:
                    out.append(f"\\u{code:04x}")
                continue

            out.append(ch)
            continue

        if ch == '"':
            in_string = True

        out.append(ch)

    return "".join(out)


def try_json_loads(text: str) -> Optional[Any]:
    """Try direct JSON parsing, repaired JSON parsing, then JSON substring parsing."""
    candidates = [text, repair_json_control_chars(text)]

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, str):
                try:
                    data = json.loads(data)
                except json.JSONDecodeError:
                    pass
            return data
        except json.JSONDecodeError:
            pass

    # LLMs sometimes add a short prose prefix/suffix around valid JSON.
    repaired = repair_json_control_chars(text)
    decoder = json.JSONDecoder()

    first_positions = [
        pos for pos in (
            repaired.find("["),
            repaired.find("{"),
        )
        if pos >= 0
    ]

    for start in sorted(first_positions):
        try:
            data, _ = decoder.raw_decode(repaired[start:])
            return data
        except json.JSONDecodeError:
            continue

    return None


def parse_articles(value: Any) -> List[Dict[str, str]]:
    """
    Parse an LLM response into a list of {title, body} dictionaries.

    Supported top-level forms:
        [ {"title": ..., "body": ...}, ... ]
        {"title": ..., "body": ...}
        {"articles": [ ... ]}

    Invalid/missing values return an empty list.
    """
    if value is None:
        return []

    if isinstance(value, list):
        data = value
    elif isinstance(value, dict):
        data = value
    elif is_missing(value):
        return []
    else:
        raw = safe_text(value).strip()
        if not raw:
            return []

        raw = strip_markdown_fence(raw)
        data = try_json_loads(raw)

        if data is None:
            return []

    if isinstance(data, dict) and "articles" in data:
        data = data["articles"]

    if isinstance(data, dict):
        data = [data]

    if not isinstance(data, list):
        return []

    articles: List[Dict[str, str]] = []

    for item in data:
        if not isinstance(item, dict):
            continue

        title = item.get("title", "")
        body = item.get(
            "body",
            item.get(
                "text",
                item.get("content", ""),
            ),
        )

        articles.append({
            "title": safe_text(title),
            "body": safe_text(body),
        })

    return articles


def parse_was_failure(value: Any, parsed_articles: List[Dict[str, str]]) -> bool:
    """Return True when a non-empty response failed to yield any article."""
    if parsed_articles:
        return False

    if value is None or is_missing(value):
        return False

    raw = safe_text(value).strip()
    if not raw:
        return False

    stripped = strip_markdown_fence(raw)
    if stripped in {"[]", "{}", 'null', '""'}:
        return False

    return True


# =========================================================
# Metrics
# =========================================================

def _levenshtein(a: List[Any], b: List[Any]) -> int:
    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    prev = list(range(len(b) + 1))

    for i, ca in enumerate(a, start=1):
        curr = [i] + [0] * len(b)

        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1

            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + cost,
            )

        prev = curr

    return prev[-1]


def cer(hyp: str, ref: str) -> float:
    """Character Error Rate."""
    hyp = safe_text(hyp)
    ref = safe_text(ref)

    if not ref:
        return 0.0 if not hyp else 1.0

    return _levenshtein(
        list(hyp),
        list(ref),
    ) / len(ref)


def wer(hyp: str, ref: str) -> float:
    """Word Error Rate."""
    hyp = safe_text(hyp)
    ref = safe_text(ref)

    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return _levenshtein(
        hyp_words,
        ref_words,
    ) / len(ref_words)


def similarity_score(
    pred_title: str,
    pred_body: str,
    gold_title: str,
    gold_body: str,
) -> Tuple[float, float, float]:
    """
    Weighted rapidfuzz similarity (0-100).

    Returns:
        combined_score,
        title_score,
        body_score
    """
    pred_title_n = normalize_text(pred_title)
    pred_body_n = normalize_text(pred_body)
    gold_title_n = normalize_text(gold_title)
    gold_body_n = normalize_text(gold_body)

    title_score = float(
        ratio(pred_title_n, gold_title_n)
    )

    body_score = float(
        ratio(pred_body_n, gold_body_n)
    )

    combined = (
        TITLE_WEIGHT * title_score
        + BODY_WEIGHT * body_score
    )

    return combined, title_score, body_score


# =========================================================
# Timing
# =========================================================

def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    """Numeric timing column with invalid values converted to NaN."""
    return pd.to_numeric(
        df[column],
        errors="coerce",
    ).astype(float)


def build_timing(
    results_df: pd.DataFrame,
    enhance_df: pd.DataFrame,
    binarize_df: pd.DataFrame,
    ocr_df: pd.DataFrame,
    layout_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Join per-stage timing onto results_df.

    Stage join keys:

        enhance:
            image_stem + config_id

        layout:
            image_stem + config_id + detector

        binarize:
            image_stem + config_id + detector + binarization

        ocr:
            image_stem + config_id + layout_mode + detector
            + binarization + ocr_extractor

        llm:
            COMBO_COLS; max elapsed_s is used so one LLM execution is
            counted once even when duplicate result rows exist.
    """
    df = results_df.copy()

    if "detector" not in df.columns:
        df["detector"] = DETECTOR_NONE

    df["detector"] = df["detector"].fillna(DETECTOR_NONE)
    df["elapsed_s"] = _numeric_series(df, "elapsed_s")

    # -----------------------------------------------------
    # Enhancement
    # -----------------------------------------------------
    enh = enhance_df.copy()
    enh["image_stem"] = enh["image_path"].apply(
        lambda p: Path(p).stem
    )
    enh["processing_time_seconds"] = _numeric_series(
        enh,
        "processing_time_seconds",
    )

    enh_agg = (
        enh.groupby(
            ["image_stem", "config_id"],
            dropna=False,
        )["processing_time_seconds"]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={
                "processing_time_seconds": "_t_enhance",
            }
        )
    )

    df = df.merge(
        enh_agg,
        on=["image_stem", "config_id"],
        how="left",
    )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------
    if layout_df is not None:
        lay = layout_df.copy()
        lay["_det"] = lay["detector"].fillna(DETECTOR_NONE)
        lay["elapsed_s"] = _numeric_series(lay, "elapsed_s")

        lay_agg = (
            lay.groupby(
                [
                    "image_stem",
                    "config_id",
                    "_det",
                ],
                dropna=False,
            )["elapsed_s"]
            .sum(min_count=1)
            .reset_index()
            .rename(
                columns={
                    "elapsed_s": "_t_layout",
                }
            )
        )

        df = df.merge(
            lay_agg,
            left_on=[
                "image_stem",
                "config_id",
                "detector",
            ],
            right_on=[
                "image_stem",
                "config_id",
                "_det",
            ],
            how="left",
        )
        df.drop(columns=["_det"], inplace=True, errors="ignore")
    else:
        df["_t_layout"] = 0.0

    # -----------------------------------------------------
    # Binarization
    # -----------------------------------------------------
    bz = binarize_df.copy()
    bz["_det"] = bz["detector"].fillna(DETECTOR_NONE)
    bz["elapsed_s"] = _numeric_series(bz, "elapsed_s")

    bz_agg = (
        bz.groupby(
            [
                "image_stem",
                "config_id",
                "_det",
                "binarization",
            ],
            dropna=False,
        )["elapsed_s"]
        .mean()
        .reset_index()
        .rename(
            columns={
                "elapsed_s": "_t_binarize",
            }
        )
    )

    df = df.merge(
        bz_agg,
        left_on=[
            "image_stem",
            "config_id",
            "detector",
            "binarization",
        ],
        right_on=[
            "image_stem",
            "config_id",
            "_det",
            "binarization",
        ],
        how="left",
    )
    df.drop(columns=["_det"], inplace=True, errors="ignore")

    # -----------------------------------------------------
    # OCR
    # -----------------------------------------------------
    ocr = ocr_df.copy()
    ocr["_det"] = ocr["detector"].fillna(DETECTOR_NONE)
    ocr["elapsed_s"] = _numeric_series(ocr, "elapsed_s")

    ocr_agg = (
        ocr.groupby(
            [
                "image_stem",
                "config_id",
                "layout_mode",
                "_det",
                "binarization",
                "ocr_extractor",
            ],
            dropna=False,
        )["elapsed_s"]
        .sum(min_count=1)
        .reset_index()
        .rename(
            columns={
                "elapsed_s": "_t_ocr",
            }
        )
    )

    df = df.merge(
        ocr_agg,
        left_on=[
            "image_stem",
            "config_id",
            "layout_mode",
            "detector",
            "binarization",
            "ocr_extractor",
        ],
        right_on=[
            "image_stem",
            "config_id",
            "layout_mode",
            "_det",
            "binarization",
            "ocr_extractor",
        ],
        how="left",
    )
    df.drop(columns=["_det"], inplace=True, errors="ignore")

    # -----------------------------------------------------
    # LLM: one execution per combo
    # -----------------------------------------------------
    llm_agg = (
        df.groupby(
            COMBO_COLS,
            dropna=False,
        )["elapsed_s"]
        .max()
        .reset_index()
        .rename(
            columns={
                "elapsed_s": "_t_llm",
            }
        )
    )

    df.drop(columns=["elapsed_s"], inplace=True, errors="ignore")

    df = df.merge(
        llm_agg,
        on=COMBO_COLS,
        how="left",
    )

    # -----------------------------------------------------
    # Total pipeline time
    # -----------------------------------------------------
    for col in (
        "_t_enhance",
        "_t_layout",
        "_t_binarize",
        "_t_ocr",
        "_t_llm",
    ):
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce",
        ).fillna(0.0)

    df["total_pipeline_seconds"] = (
        df["_t_enhance"]
        + df["_t_layout"]
        + df["_t_binarize"]
        + df["_t_ocr"]
        + df["_t_llm"]
    )

    df.drop(
        columns=[
            "_t_enhance",
            "_t_layout",
            "_t_binarize",
            "_t_ocr",
            "_t_llm",
        ],
        inplace=True,
        errors="ignore",
    )

    return df


def page_pipeline_time(combo_df: pd.DataFrame) -> Tuple[float, bool]:
    """
    Return one execution time for a combo.

    Normally there is one value. If duplicate rows still contain slightly or
    substantially different values, use the maximum rather than multiplying
    the execution time by the number of returned articles/rows.

    Returns:
        total_seconds,
        timing_was_ambiguous
    """
    values = pd.to_numeric(
        combo_df["total_pipeline_seconds"],
        errors="coerce",
    ).dropna()

    if values.empty:
        return 0.0, False

    unique_values = np.unique(
        np.round(values.to_numpy(dtype=float), 9)
    )

    if len(unique_values) == 1:
        return float(unique_values[0]), False

    return float(values.max()), True


# =========================================================
# Evaluation helpers
# =========================================================

def make_gold_lookup(gold_df: pd.DataFrame) -> Dict[Any, List[Dict[str, str]]]:
    """Pre-group gold articles by image stem."""
    gold = gold_df.copy()
    gold["_title"] = gold["title"].map(normalize_text)
    gold["_body"] = gold["body"].map(normalize_text)

    lookup: Dict[Any, List[Dict[str, str]]] = {}

    for image_stem, group in gold.groupby(
        "image_stem",
        dropna=False,
        sort=False,
    ):
        rows: List[Dict[str, str]] = []

        for row in group.itertuples(index=False):
            rows.append({
                "title": safe_text(getattr(row, "title", "")),
                "body": safe_text(getattr(row, "body", "")),
                "_title": safe_text(getattr(row, "_title", "")),
                "_body": safe_text(getattr(row, "_body", "")),
            })

        try:
            lookup[image_stem] = rows
        except TypeError:
            pass

    return lookup


def get_gold_for_image(
    gold_lookup: Dict[Any, List[Dict[str, str]]],
    image_stem: Any,
) -> List[Dict[str, str]]:
    """Fetch gold rows for an image stem."""
    try:
        return gold_lookup.get(image_stem, [])
    except TypeError:
        return []


def write_article_row(
    writer: csv.DictWriter,
    row: Dict[str, Any],
) -> None:
    """Write one article-level row in the declared column order."""
    clean = {
        key: csv_value(row.get(key))
        for key in ARTICLE_FIELDS
    }
    writer.writerow(clean)


def write_page_row(
    writer: csv.DictWriter,
    row: Dict[str, Any],
) -> None:
    """Write one page-level row in the declared column order."""
    clean = {
        key: csv_value(row.get(key))
        for key in PAGE_FIELDS
    }
    writer.writerow(clean)


def audit_print(
    combo_count: int,
    meta: Dict[str, Any],
    audit_rows: List[Dict[str, Any]],
) -> None:
    """Print one random audit sample."""
    print()
    print(f"--- random audit after {combo_count:,} successful combinations ---")
    print(
        f"[{meta['image_stem']}] "
        f"config={meta['config_id']} | "
        f"layout={meta['layout_mode']} | "
        f"detector={meta['detector']} | "
        f"binarization={meta['binarization']} | "
        f"ocr={meta['ocr_extractor']} | "
        f"llm={meta['llm_extractor']}"
    )

    if not audit_rows:
        print("No alignable article pair in this combination.")
        return

    sample = random.choice(audit_rows)

    print(f"annotation: {sample['result_type']}")
    print(
        "matching_score: "
        f"{sample['matching_score']:.4f}"
    )
    print(
        "title_fuzzy_score: "
        f"{sample['title_fuzzy_score']:.4f}"
    )
    print(
        "body_fuzzy_score: "
        f"{sample['body_fuzzy_score']:.4f}"
    )
    print(f"pred_title: {preview(sample['pred_title'])}")
    print(f"pred_body:  {preview(sample['pred_body'])}")
    print(f"gold_title: {preview(sample['gold_title'])}")
    print(f"gold_body:  {preview(sample['gold_body'])}")
    print()


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    results_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    article_output_path: Path,
    page_output_path: Path,
) -> Tuple[int, int, int, int]:
    """
    Evaluate all successful pipeline combinations and stream both CSVs.

    Returns:
        article_row_count,
        page_row_count,
        successful_combo_count,
        timing_ambiguous_combo_count
    """
    article_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    page_output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    # ---------------------------------------------------------
    # Normalize detector and status
    # ---------------------------------------------------------
    results = results_df.copy()

    if "detector" not in results.columns:
        results["detector"] = DETECTOR_NONE
    results["detector"] = results["detector"].fillna(DETECTOR_NONE)

    results["_status"] = (
        results["status"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # ---------------------------------------------------------
    # Console status: exactly one line per unique image stem
    # ---------------------------------------------------------
    status_counts = (
        results.groupby(
            "image_stem",
            dropna=False,
            sort=True,
        )["_status"]
        .value_counts()
        .unstack(fill_value=0)
    )

    all_images = list(status_counts.index)

    for image_stem in all_images:
        row = status_counts.loc[image_stem]
        n_success = int(row.get("success", 0))
        n_failure = int(row.get("failure", 0))

        print(
            f"[{image_stem}] "
            f"success={n_success:,} | "
            f"failure={n_failure:,}"
        )

    # ---------------------------------------------------------
    # Successful rows only for evaluation
    # ---------------------------------------------------------
    successful_results = results[
        results["_status"] == "success"
    ].copy()

    successful_combo_count = int(
        successful_results.groupby(
            COMBO_COLS,
            dropna=False,
        ).ngroups
    )

    gold_lookup = make_gold_lookup(gold_df)

    # ---------------------------------------------------------
    # Stream article + page CSVs
    # ---------------------------------------------------------
    article_row_count = 0
    page_row_count = 0
    timing_ambiguous_combo_count = 0
    processed_combos = 0

    pbar = tqdm(
        total=successful_combo_count,
        desc="Evaluating successful pipeline combinations",
        leave=True,
        unit="combo",
    )

    try:
        with article_output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as article_handle, page_output_path.open(
            "w",
            newline="",
            encoding="utf-8",
        ) as page_handle:
            article_writer = csv.DictWriter(
                article_handle,
                fieldnames=ARTICLE_FIELDS,
                extrasaction="ignore",
            )
            page_writer = csv.DictWriter(
                page_handle,
                fieldnames=PAGE_FIELDS,
                extrasaction="ignore",
            )

            article_writer.writeheader()
            page_writer.writeheader()

            # Iterate image-by-image to keep working memory bounded.
            for image_stem, image_results in successful_results.groupby(
                "image_stem",
                dropna=False,
                sort=False,
            ):
                image_gold = get_gold_for_image(
                    gold_lookup,
                    image_stem,
                )
                n_gold = len(image_gold)

                image_combos = image_results.groupby(
                    COMBO_COLS,
                    dropna=False,
                    sort=False,
                )

                for combo, combo_df in image_combos:
                    (
                        _image_stem,
                        config_id,
                        layout_mode,
                        detector,
                        binarization,
                        ocr_extractor,
                        llm_extractor,
                    ) = combo

                    meta = combo_meta(
                        image_stem=image_stem,
                        config_id=config_id,
                        layout_mode=layout_mode,
                        detector=detector,
                        binarization=binarization,
                        ocr_extractor=ocr_extractor,
                        llm_extractor=llm_extractor,
                    )

                    # -------------------------------------------------
                    # One execution time for the whole combo
                    # -------------------------------------------------
                    total_pipeline_seconds, timing_ambiguous = (
                        page_pipeline_time(combo_df)
                    )

                    if timing_ambiguous:
                        timing_ambiguous_combo_count += 1

                    # -------------------------------------------------
                    # Parse all LLM outputs in this successful combo
                    # -------------------------------------------------
                    pred_list: List[Dict[str, str]] = []
                    parse_failure_rows = 0

                    for raw_body in combo_df["body"]:
                        parsed = parse_articles(raw_body)
                        if parse_was_failure(raw_body, parsed):
                            parse_failure_rows += 1
                        pred_list.extend(parsed)

                    n_pred = len(pred_list)

                    matched_pred = set()
                    tp = 0

                    # Store only alignable rows for the audit sample.
                    audit_rows: List[Dict[str, Any]] = []

                    # -------------------------------------------------
                    # Hungarian matching
                    # -------------------------------------------------
                    if n_gold > 0 and n_pred > 0:
                        sim = np.zeros(
                            (n_gold, n_pred),
                            dtype=float,
                        )
                        title_sim = np.zeros_like(sim)
                        body_sim = np.zeros_like(sim)

                        for gi, grow in enumerate(image_gold):
                            gold_title = grow["title"]
                            gold_body = grow["body"]

                            for pi, prow in enumerate(pred_list):
                                score, title_score, body_score = similarity_score(
                                    prow["title"],
                                    prow["body"],
                                    gold_title,
                                    gold_body,
                                )

                                sim[gi, pi] = score
                                title_sim[gi, pi] = title_score
                                body_sim[gi, pi] = body_score

                        gold_idx, pred_idx = linear_sum_assignment(
                            100.0 - sim
                        )

                        for gi, pi in zip(gold_idx, pred_idx):
                            score = float(sim[gi, pi])

                            if score < MIN_MATCH_SCORE:
                                continue

                            matched_pred.add(pi)

                            grow = image_gold[gi]
                            prow = pred_list[pi]

                            pred_title = prow["title"]
                            pred_body = prow["body"]
                            gold_title = grow["title"]
                            gold_body = grow["body"]

                            is_tp = score >= MATCH_THRESHOLD
                            result_type = "TP" if is_tp else "FP"

                            if is_tp:
                                tp += 1

                            article_row = {
                                **meta,
                                "result_type": result_type,
                                "pred_title": pred_title,
                                "pred_body": pred_body,
                                "gold_title": gold_title,
                                "gold_body": gold_body,
                                "matching_score": round(score, 4),
                                "title_fuzzy_score": round(
                                    float(title_sim[gi, pi]),
                                    4,
                                ),
                                "body_fuzzy_score": round(
                                    float(body_sim[gi, pi]),
                                    4,
                                ),
                                "title_cer": cer(
                                    pred_title,
                                    gold_title,
                                ),
                                "title_wer": wer(
                                    pred_title,
                                    gold_title,
                                ),
                                "body_cer": cer(
                                    pred_body,
                                    gold_body,
                                ),
                                "body_wer": wer(
                                    pred_body,
                                    gold_body,
                                ),
                                "body_len_delta_chars": (
                                    len(pred_body) - len(gold_body)
                                ),
                                "total_pipeline_seconds": (
                                    total_pipeline_seconds
                                ),
                            }

                            write_article_row(
                                article_writer,
                                article_row,
                            )
                            article_row_count += 1

                            audit_rows.append(article_row)

                    # -------------------------------------------------
                    # Unmatched predictions = FP
                    # -------------------------------------------------
                    for pi, prow in enumerate(pred_list):
                        if pi in matched_pred:
                            continue

                        article_row = {
                            **meta,
                            "result_type": "FP",
                            "pred_title": prow["title"],
                            "pred_body": prow["body"],
                            "gold_title": None,
                            "gold_body": None,
                            "matching_score": None,
                            "title_fuzzy_score": None,
                            "body_fuzzy_score": None,
                            "title_cer": None,
                            "title_wer": None,
                            "body_cer": None,
                            "body_wer": None,
                            "body_len_delta_chars": None,
                            "total_pipeline_seconds": (
                                total_pipeline_seconds
                            ),
                        }

                        write_article_row(
                            article_writer,
                            article_row,
                        )
                        article_row_count += 1

                    # -------------------------------------------------
                    # FN semantics
                    # -------------------------------------------------
                    # Preserve the agreed semantics:
                    # every gold article not detected as TP counts as FN.
                    # Thus a borderline alignable match (20-69.999) is both
                    # an FP prediction and a missed (FN) gold article.
                    fn = n_gold - tp

                    # A gold article is an FN row whenever it was not assigned
                    # a true-positive prediction. Borderline matches therefore
                    # remain present as FN rows, while true TP gold articles do not.
                    tp_gold = set()
                    if n_gold > 0 and n_pred > 0:
                        for gi, pi in zip(gold_idx, pred_idx):
                            if sim[gi, pi] >= MATCH_THRESHOLD:
                                tp_gold.add(int(gi))

                    fn_rows_written = 0

                    for gi, grow in enumerate(image_gold):
                        if gi in tp_gold:
                            continue

                        article_row = {
                            **meta,
                            "result_type": "FN",
                            "pred_title": None,
                            "pred_body": None,
                            "gold_title": grow["title"],
                            "gold_body": grow["body"],
                            "matching_score": None,
                            "title_fuzzy_score": None,
                            "body_fuzzy_score": None,
                            "title_cer": None,
                            "title_wer": None,
                            "body_cer": None,
                            "body_wer": None,
                            "body_len_delta_chars": None,
                            "total_pipeline_seconds": (
                                total_pipeline_seconds
                            ),
                        }

                        write_article_row(
                            article_writer,
                            article_row,
                        )
                        article_row_count += 1
                        fn_rows_written += 1

                    # In the unlikely event there are more FN rows implied by the
                    # count than unique non-TP gold rows (normally impossible),
                    # preserve the count without duplicating useful text.
                    while fn_rows_written < fn:
                        article_row = {
                            **meta,
                            "result_type": "FN",
                            "pred_title": None,
                            "pred_body": None,
                            "gold_title": None,
                            "gold_body": None,
                            "matching_score": None,
                            "title_fuzzy_score": None,
                            "body_fuzzy_score": None,
                            "title_cer": None,
                            "title_wer": None,
                            "body_cer": None,
                            "body_wer": None,
                            "body_len_delta_chars": None,
                            "total_pipeline_seconds": (
                                total_pipeline_seconds
                            ),
                        }

                        write_article_row(
                            article_writer,
                            article_row,
                        )
                        article_row_count += 1
                        fn_rows_written += 1

                    # -------------------------------------------------
                    # Detection counts
                    # -------------------------------------------------
                    # Every prediction that is not a TP is an FP. This makes
                    # borderline alignable matches (20-69.999) count as FP.
                    fp = n_pred - tp

                    precision = (
                        tp / (tp + fp)
                        if (tp + fp)
                        else 0.0
                    )

                    recall = (
                        tp / (tp + fn)
                        if (tp + fn)
                        else 0.0
                    )

                    detection_f1 = (
                        2.0 * precision * recall
                        / (precision + recall)
                        if (precision + recall)
                        else 0.0
                    )

                    # -------------------------------------------------
                    # Text quality: matched/aligned pairs only
                    # -------------------------------------------------
                    matched_title_cer: List[float] = []
                    matched_title_wer: List[float] = []
                    matched_body_cer: List[float] = []
                    matched_body_wer: List[float] = []

                    if n_gold > 0 and n_pred > 0:
                        for gi, pi in zip(gold_idx, pred_idx):
                            score = float(sim[gi, pi])
                            if score < MIN_MATCH_SCORE:
                                continue

                            grow = image_gold[gi]
                            prow = pred_list[pi]

                            matched_title_cer.append(
                                cer(
                                    prow["title"],
                                    grow["title"],
                                )
                            )
                            matched_title_wer.append(
                                wer(
                                    prow["title"],
                                    grow["title"],
                                )
                            )
                            matched_body_cer.append(
                                cer(
                                    prow["body"],
                                    grow["body"],
                                )
                            )
                            matched_body_wer.append(
                                wer(
                                    prow["body"],
                                    grow["body"],
                                )
                            )

                    page_row = {
                        **meta,
                        "gold_articles": n_gold,
                        "predicted_articles": n_pred,
                        "tp": tp,
                        "fp": fp,
                        "fn": fn,
                        "detection_precision": round(precision, 6),
                        "detection_recall": round(recall, 6),
                        "detection_f1": round(detection_f1, 6),
                        "matched_pairs_for_error_mean": len(matched_title_cer),
                        "matched_title_cer_mean": (
                            float(np.mean(matched_title_cer))
                            if matched_title_cer
                            else None
                        ),
                        "matched_title_wer_mean": (
                            float(np.mean(matched_title_wer))
                            if matched_title_wer
                            else None
                        ),
                        "matched_body_cer_mean": (
                            float(np.mean(matched_body_cer))
                            if matched_body_cer
                            else None
                        ),
                        "matched_body_wer_mean": (
                            float(np.mean(matched_body_wer))
                            if matched_body_wer
                            else None
                        ),
                        "total_pipeline_seconds": total_pipeline_seconds,
                        "parse_failure_rows": parse_failure_rows,
                    }

                    write_page_row(
                        page_writer,
                        page_row,
                    )
                    page_row_count += 1

                    processed_combos += 1
                    pbar.update(1)

                    # -------------------------------------------------
                    # Random audit every N successful combinations
                    # -------------------------------------------------
                    if processed_combos % AUDIT_EVERY == 0:
                        audit_print(
                            processed_combos,
                            meta,
                            audit_rows,
                        )

                    # Flush regularly without flushing on every row.
                    if processed_combos % 25 == 0:
                        article_handle.flush()
                        page_handle.flush()
    finally:
        pbar.close()

    return (
        article_row_count,
        page_row_count,
        processed_combos,
        timing_ambiguous_combo_count,
    )


# =========================================================
# CLI
# =========================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate OCR/LLM extraction results."
    )

    parser.add_argument(
        "--results",
        required=True,
        help="LLM extraction results parquet.",
    )

    parser.add_argument(
        "--gold",
        required=True,
        help="Gold-standard CSV (image_stem, title, body).",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output article-level evaluation CSV.",
    )

    parser.add_argument(
        "--enhance-parquet",
        required=True,
        help="Enhancement timing parquet.",
    )

    parser.add_argument(
        "--binarize-parquet",
        required=True,
        help="Binarization timing parquet.",
    )

    parser.add_argument(
        "--ocr-parquet",
        required=True,
        help="OCR extraction timing parquet.",
    )

    parser.add_argument(
        "--layout-parquet",
        default=None,
        help="Layout detection timing parquet (optional).",
    )

    return parser.parse_args()


# =========================================================
# Main
# =========================================================

def main():
    args = parse_args()

    # -----------------------------------------------------
    # Load data
    # -----------------------------------------------------
    results_df = pd.read_parquet(args.results)

    gold_df = pd.read_csv(
        args.gold,
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        encoding="utf-8",
    )

    enhance_df = pd.read_parquet(args.enhance_parquet)
    binarize_df = pd.read_parquet(args.binarize_parquet)
    ocr_df = pd.read_parquet(args.ocr_parquet)

    layout_df = (
        pd.read_parquet(args.layout_parquet)
        if args.layout_parquet
        else None
    )

    # -----------------------------------------------------
    # Build timing
    # -----------------------------------------------------
    results_df = build_timing(
        results_df,
        enhance_df,
        binarize_df,
        ocr_df,
        layout_df,
    )

    # -----------------------------------------------------
    # Output paths
    # -----------------------------------------------------
    output_path = Path(args.output_csv)

    page_stem = output_path.stem
    if page_stem.endswith("_results"):
        page_stem = page_stem[:-len("_results")]

    page_output = output_path.with_name(
        f"{page_stem}_page_metrics.csv"
    )

    # -----------------------------------------------------
    # Evaluate + stream outputs
    # -----------------------------------------------------
    (
        article_count,
        page_count,
        combo_count,
        timing_ambiguous_count,
    ) = evaluate(
        results_df,
        gold_df,
        output_path,
        page_output,
    )

    print()
    print(f"Article metrics -> {output_path}")
    print(f"Page metrics    -> {page_output}")
    print(
        f"Total rows: article={article_count:,}, "
        f"page={page_count:,}"
    )
    print(
        f"Successful pipeline combinations evaluated: "
        f"{combo_count:,}"
    )

    if timing_ambiguous_count:
        print(
            "Timing note: "
            f"{timing_ambiguous_count:,} combo(s) contained multiple "
            "different total_pipeline_seconds values; the evaluator used "
            "the maximum once per combo rather than summing duplicate rows."
        )


if __name__ == "__main__":
    main()