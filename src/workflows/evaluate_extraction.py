"""
evaluate_extraction.py

Evaluate OCR/VLM extraction results against a gold-standard CSV.

Gold CSV:

    image_stem,title,body

For each available:

    (config_id, detector, binarize_file, vlm)

combination and each gold image:

1. Restrict predictions to the same image_stem.
2. Consider only rows with status == "success".
3. Normalize title/body text.
4. Compute pairwise article similarity:
       0.3 * title token F1 + 0.7 * body token F1
5. Find the globally optimal one-to-one assignment using
   the Hungarian algorithm.
6. Accept assignments only when similarity >= MIN_MATCH_SCORE.
7. Compute field-level fuzzy similarity and conditional CER/WER.
8. Emit one row for every matched gold article, unmatched gold
   article (FN), and unmatched prediction (FP).

Page-level detection metrics are also calculated internally.

Important:
    total_pipeline_seconds is the runtime associated with the
    image/configuration/pipeline. It is repeated on article rows
    belonging to the same pipeline execution and therefore should
    NOT be summed across article rows.
"""

import argparse
import csv
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
from rapidfuzz.fuzz import ratio
from scipy.optimize import linear_sum_assignment


# =========================================================
# Configuration
# =========================================================

COMBO_COLS = [
    "config_id",
    "detector",
    "binarize_file",
    "vlm",
]

TITLE_FUZZY_THRESHOLD = 75
BODY_FUZZY_THRESHOLD = 75
MIN_MATCH_SCORE = 0.20

TITLE_MATCH_WEIGHT = 0.30
BODY_MATCH_WEIGHT = 0.70

DETECTOR_NONE = "__none__"


# =========================================================
# Fuzzy matching
# =========================================================

def fuzzy_score(hyp: str, ref: str) -> float:
    """
    RapidFuzz similarity in the range [0, 100].
    """

    if not hyp and not ref:
        return 100.0

    if not hyp or not ref:
        return 0.0

    return float(ratio(hyp, ref))


def conditional_error_metric(
    hyp: str,
    ref: str,
    matched: bool,
    metric_fn,
):
    """
    Compute an error metric only when the corresponding field
    passes its fuzzy-match threshold.
    """

    if not matched:
        return None

    return metric_fn(hyp, ref)


# =========================================================
# Normalization
# =========================================================

def normalize_text(text) -> str:
    """
    Normalize text for evaluation.

    Operations:
        - null -> ""
        - Unicode NFKD normalization
        - remove combining marks
        - lowercase
        - remove punctuation
        - collapse whitespace
    """

    if text is None or pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.lower()

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text).strip()

    return text


# =========================================================
# Levenshtein
# =========================================================

def _levenshtein(a: List, b: List) -> int:
    """
    Compute Levenshtein edit distance between two sequences.
    """

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
    """
    Character Error Rate.

    Both strings are expected to already be normalized.
    """

    if not ref:
        return 0.0 if not hyp else 1.0

    return _levenshtein(
        list(hyp),
        list(ref),
    ) / len(ref)


def wer(hyp: str, ref: str) -> float:
    """
    Word Error Rate.

    Both strings are expected to already be normalized.
    """

    ref_words = ref.split()
    hyp_words = hyp.split()

    if not ref_words:
        return 0.0 if not hyp_words else 1.0

    return _levenshtein(
        hyp_words,
        ref_words,
    ) / len(ref_words)


# =========================================================
# Token F1
# =========================================================

def token_f1(
    hyp: str,
    ref: str,
) -> Tuple[float, float, float]:
    """
    Multiset token precision, recall and F1.
    """

    hyp_tokens = hyp.split()
    ref_tokens = ref.split()

    if not hyp_tokens and not ref_tokens:
        return 1.0, 1.0, 1.0

    if not hyp_tokens or not ref_tokens:
        return 0.0, 0.0, 0.0

    common = Counter(hyp_tokens) & Counter(ref_tokens)

    overlap = sum(common.values())

    if overlap == 0:
        return 0.0, 0.0, 0.0

    precision = overlap / len(hyp_tokens)
    recall = overlap / len(ref_tokens)

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    return precision, recall, f1


def article_similarity(
    pred_title: str,
    pred_body: str,
    gold_title: str,
    gold_body: str,
) -> Tuple[float, float, float]:
    """
    Calculate article identity similarity.

    Returns:
        article_score
        title_f1
        body_f1
    """

    _, _, title_f1 = token_f1(
        pred_title,
        gold_title,
    )

    _, _, body_f1 = token_f1(
        pred_body,
        gold_body,
    )

    score = (
        TITLE_MATCH_WEIGHT * title_f1
        + BODY_MATCH_WEIGHT * body_f1
    )

    return score, title_f1, body_f1


# =========================================================
# Input validation
# =========================================================

def validate_columns(
    df: pd.DataFrame,
    required: List[str],
    name: str,
):
    """
    Fail early when expected columns are missing.
    """

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"{name} is missing required columns: {missing}"
        )


# =========================================================
# Evaluation
# =========================================================

def evaluate(
    results_df: pd.DataFrame,
    gold_df: pd.DataFrame,
    image_stem_col: str = "image_stem",
    title_col: str = "title",
    body_col: str = "body",
):
    """
    Evaluate predictions against gold articles.

    Returns:
        article_df
        page_df
    """

    validate_columns(
        results_df,
        COMBO_COLS
        + [
            "status",
            "image_stem",
            "title",
            "body",
        ],
        "results",
    )

    validate_columns(
        gold_df,
        [
            image_stem_col,
            title_col,
            body_col,
        ],
        "gold",
    )

    article_rows = []
    page_rows = []

    gold = gold_df.copy()

    gold["_title"] = gold[title_col].map(normalize_text)
    gold["_body"] = gold[body_col].map(normalize_text)

    # -----------------------------------------------------
    # Normalize combo columns consistently.
    #
    # NaN values are converted to a sentinel because pandas
    # groupby/merge handling of NaN can otherwise cause
    # combinations to disappear.
    # -----------------------------------------------------

    results = results_df.copy()

    for col in ["detector", "binarize_file"]:
        if col in results.columns:
            results[col] = results[col].fillna(DETECTOR_NONE)

    # dropna=False is important here.
    combo_groups = results.groupby(
        COMBO_COLS,
        dropna=False,
    )

    for combo, combo_df in combo_groups:

        (
            config_id,
            detector,
            binarize_file,
            vlm,
        ) = combo

        # -------------------------------------------------
        # Evaluate EVERY gold image for this combination.
        #
        # This is important: if there are zero successful
        # predictions, the image still needs FN rows.
        # -------------------------------------------------

        for image_stem, image_gold in gold.groupby(
            image_stem_col,
            dropna=False,
        ):

            candidates = combo_df[
                (combo_df["status"] == "success")
                &
                (combo_df["image_stem"] == image_stem)
            ].copy()

            candidates["_title"] = candidates["title"].map(
                normalize_text
            )

            candidates["_body"] = candidates["body"].map(
                normalize_text
            )

            gold_articles = [
                (
                    idx,
                    row["_title"],
                    row["_body"],
                )
                for idx, row in image_gold.iterrows()
            ]

            pred_articles = [
                (
                    idx,
                    row,
                )
                for idx, row in candidates.iterrows()
            ]

            n_gold = len(gold_articles)
            n_pred = len(pred_articles)

            matched_predictions = set()
            matched_gold = set()

            # -------------------------------------------------
            # Match gold articles to predictions
            # -------------------------------------------------

            if n_gold > 0 and n_pred > 0:

                similarity = np.zeros(
                    (n_gold, n_pred),
                    dtype=float,
                )

                title_f1_matrix = np.zeros_like(similarity)
                body_f1_matrix = np.zeros_like(similarity)

                for gi, (_, gold_title, gold_body) in enumerate(
                    gold_articles
                ):

                    for pi, (_, pred_row) in enumerate(
                        pred_articles
                    ):

                        (
                            score,
                            title_f1,
                            body_f1,
                        ) = article_similarity(
                            pred_row["_title"],
                            pred_row["_body"],
                            gold_title,
                            gold_body,
                        )

                        similarity[gi, pi] = score
                        title_f1_matrix[gi, pi] = title_f1
                        body_f1_matrix[gi, pi] = body_f1

                # Hungarian algorithm minimizes cost.
                cost = 1.0 - similarity

                assigned_gold, assigned_predictions = (
                    linear_sum_assignment(cost)
                )

                for gi, pi in zip(
                    assigned_gold,
                    assigned_predictions,
                ):

                    score = similarity[gi, pi]

                    # Assignment exists, but it is not accepted
                    # as a true article match unless it clears
                    # the minimum similarity threshold.
                    if score < MIN_MATCH_SCORE:
                        continue

                    matched_gold.add(gi)
                    matched_predictions.add(pi)

                    gold_row = image_gold.iloc[gi]
                    pred_row = pred_articles[pi][1]

                    pred_title = pred_row["_title"]
                    pred_body = pred_row["_body"]

                    gold_title = gold_row["_title"]
                    gold_body = gold_row["_body"]

                    title_f1 = title_f1_matrix[gi, pi]
                    body_f1 = body_f1_matrix[gi, pi]

                    title_fuzzy = fuzzy_score(
                        pred_title,
                        gold_title,
                    )

                    body_fuzzy = fuzzy_score(
                        pred_body,
                        gold_body,
                    )

                    title_fuzzy_match = (
                        title_fuzzy >= TITLE_FUZZY_THRESHOLD
                    )

                    body_fuzzy_match = (
                        body_fuzzy >= BODY_FUZZY_THRESHOLD
                    )

                    article_rows.append({

                        "image_stem": image_stem,

                        "config_id": config_id,
                        "detector": detector,
                        "binarize_file": binarize_file,
                        "vlm": vlm,

                        # Identity/matching information
                        "match_status": "MATCHED",
                        "matching_score": score,

                        "gold_index": gold_articles[gi][0],
                        "prediction_index": pred_articles[pi][0],

                        # Field-level similarity
                        "title_token_f1": title_f1,
                        "body_token_f1": body_f1,

                        "title_fuzzy_score": title_fuzzy,
                        "title_fuzzy_match": title_fuzzy_match,

                        "body_fuzzy_score": body_fuzzy,
                        "body_fuzzy_match": body_fuzzy_match,

                        # Field-level status
                        "title_match_status": (
                            "MATCH"
                            if title_fuzzy_match
                            else "MISMATCH"
                        ),

                        "body_match_status": (
                            "MATCH"
                            if body_fuzzy_match
                            else "MISMATCH"
                        ),

                        # Conditional error metrics
                        "title_cer": conditional_error_metric(
                            pred_title,
                            gold_title,
                            title_fuzzy_match,
                            cer,
                        ),

                        "title_wer": conditional_error_metric(
                            pred_title,
                            gold_title,
                            title_fuzzy_match,
                            wer,
                        ),

                        "body_cer": conditional_error_metric(
                            pred_body,
                            gold_body,
                            body_fuzzy_match,
                            cer,
                        ),

                        "body_wer": conditional_error_metric(
                            pred_body,
                            gold_body,
                            body_fuzzy_match,
                            wer,
                        ),

                        "body_len_delta_chars": (
                            len(pred_body)
                            - len(gold_body)
                        ),

                        # Keep prediction timing if present.
                        "vlm_elapsed_s": (
                            pred_row["elapsed_s"]
                            if "elapsed_s" in pred_row.index
                            and pd.notna(pred_row["elapsed_s"])
                            else None
                        ),
                    })

            # -------------------------------------------------
            # False negatives
            # -------------------------------------------------

            for gi, (gold_idx, _, _) in enumerate(
                gold_articles
            ):

                if gi in matched_gold:
                    continue

                article_rows.append({

                    "image_stem": image_stem,

                    "config_id": config_id,
                    "detector": detector,
                    "binarize_file": binarize_file,
                    "vlm": vlm,

                    "match_status": "FN",
                    "matching_score": 0.0,

                    "gold_index": gold_idx,
                    "prediction_index": None,

                    "title_token_f1": None,
                    "body_token_f1": None,

                    "title_fuzzy_score": None,
                    "title_fuzzy_match": False,

                    "body_fuzzy_score": None,
                    "body_fuzzy_match": False,

                    "title_match_status": "FN",
                    "body_match_status": "FN",

                    "title_cer": None,
                    "title_wer": None,
                    "body_cer": None,
                    "body_wer": None,

                    "body_len_delta_chars": None,

                    "vlm_elapsed_s": None,
                })

            # -------------------------------------------------
            # False positives
            # -------------------------------------------------

            for pi, (pred_idx, _) in enumerate(
                pred_articles
            ):

                if pi in matched_predictions:
                    continue

                article_rows.append({

                    "image_stem": image_stem,

                    "config_id": config_id,
                    "detector": detector,
                    "binarize_file": binarize_file,
                    "vlm": vlm,

                    "match_status": "FP",
                    "matching_score": 0.0,

                    "gold_index": None,
                    "prediction_index": pred_idx,

                    "title_token_f1": None,
                    "body_token_f1": None,

                    "title_fuzzy_score": None,
                    "title_fuzzy_match": False,

                    "body_fuzzy_score": None,
                    "body_fuzzy_match": False,

                    "title_match_status": "FP",
                    "body_match_status": "FP",

                    "title_cer": None,
                    "title_wer": None,
                    "body_cer": None,
                    "body_wer": None,

                    "body_len_delta_chars": None,

                    "vlm_elapsed_s": (
                        candidates.loc[
                            pred_idx,
                            "elapsed_s"
                        ]
                        if "elapsed_s" in candidates.columns
                        and pred_idx in candidates.index
                        and pd.notna(
                            candidates.loc[pred_idx, "elapsed_s"]
                        )
                        else None
                    ),
                })

            # -------------------------------------------------
            # Page-level detection metrics
            # -------------------------------------------------

            matched = len(matched_gold)

            tp = matched
            fp = n_pred - matched
            fn = n_gold - matched

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
                2 * precision * recall
                / (precision + recall)
                if (precision + recall)
                else 0.0
            )

            page_rows.append({

                "image_stem": image_stem,

                "config_id": config_id,
                "detector": detector,
                "binarize_file": binarize_file,
                "vlm": vlm,

                "gold_articles": n_gold,
                "predicted_articles": n_pred,
                "matched_articles": matched,

                "tp_articles": tp,
                "fp_articles": fp,
                "fn_articles": fn,

                "detection_precision": precision,
                "detection_recall": recall,
                "detection_f1": detection_f1,
            })

    article_df = pd.DataFrame(article_rows)
    page_df = pd.DataFrame(page_rows)

    return article_df, page_df


# =========================================================
# Timing
# =========================================================

def add_timing(
    df: pd.DataFrame,
    enhance_df: pd.DataFrame,
    binarize_df: pd.DataFrame,
    layout_df: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Add pipeline timing information.

    Timing keys:

        Enhancement:
            image_stem + config_id

        Layout:
            image_stem + config_id + detector

        Binarization:
            image_stem + config_id + detector + binarize_file

        VLM:
            already associated with each prediction row

    IMPORTANT:
        Pipeline timing is image/pipeline-level timing. If the
        resulting value appears on multiple article rows, it
        must not be summed across article rows.
    """

    result = df.copy()

    # -----------------------------------------------------
    # Enhancement
    # -----------------------------------------------------

    enhance = enhance_df.copy()

    validate_columns(
        enhance,
        [
            "image_path",
            "config_id",
            "processing_time_seconds",
        ],
        "enhance parquet",
    )

    enhance["image_stem"] = enhance["image_path"].apply(
        lambda path: Path(path).stem
    )

    enhance_agg = (
        enhance
        .groupby(
            ["image_stem", "config_id"],
            dropna=False,
        )["processing_time_seconds"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "processing_time_seconds": "_enhance_s"
            }
        )
    )

    result = result.merge(
        enhance_agg,
        on=["image_stem", "config_id"],
        how="left",
    )

    # -----------------------------------------------------
    # Detector normalization
    # -----------------------------------------------------

    result["detector"] = result["detector"].fillna(
        DETECTOR_NONE
    )

    # -----------------------------------------------------
    # Layout
    # -----------------------------------------------------

    if layout_df is not None:

        layout = layout_df.copy()

        validate_columns(
            layout,
            [
                "image_stem",
                "config_id",
                "detector",
                "elapsed_s",
            ],
            "layout parquet",
        )

        layout["detector"] = layout["detector"].fillna(
            DETECTOR_NONE
        )

        layout_agg = (
            layout
            .groupby(
                [
                    "image_stem",
                    "config_id",
                    "detector",
                ],
                dropna=False,
            )["elapsed_s"]
            .sum()
            .reset_index()
            .rename(
                columns={
                    "elapsed_s": "_layout_s"
                }
            )
        )

        result = result.merge(
            layout_agg,
            on=[
                "image_stem",
                "config_id",
                "detector",
            ],
            how="left",
        )

    else:
        result["_layout_s"] = 0.0

    result["_layout_s"] = result["_layout_s"].fillna(0.0)

    # -----------------------------------------------------
    # Binarization
    # -----------------------------------------------------

    binarize = binarize_df.copy()

    validate_columns(
        binarize,
        [
            "image_stem",
            "config_id",
            "detector",
            "binarize_file",
            "elapsed_s",
        ],
        "binarize parquet",
    )

    binarize["detector"] = binarize["detector"].fillna(
        DETECTOR_NONE
    )

    binarize_agg = (
        binarize
        .groupby(
            [
                "image_stem",
                "config_id",
                "detector",
                "binarize_file",
            ],
            dropna=False,
        )["elapsed_s"]
        .sum()
        .reset_index()
        .rename(
            columns={
                "elapsed_s": "_binarize_s"
            }
        )
    )

    result = result.merge(
        binarize_agg,
        on=[
            "image_stem",
            "config_id",
            "detector",
            "binarize_file",
        ],
        how="left",
    )

    # -----------------------------------------------------
    # VLM
    # -----------------------------------------------------

    if "vlm_elapsed_s" in result.columns:
        result["_vlm_s"] = pd.to_numeric(
            result["vlm_elapsed_s"],
            errors="coerce",
        ).fillna(0.0)
    else:
        result["_vlm_s"] = 0.0

    # -----------------------------------------------------
    # Total
    # -----------------------------------------------------

    result["total_pipeline_seconds"] = (
        result["_enhance_s"].fillna(0.0)
        + result["_layout_s"].fillna(0.0)
        + result["_binarize_s"].fillna(0.0)
        + result["_vlm_s"]
    )

    # Restore missing detector values.
    result["detector"] = result["detector"].replace(
        DETECTOR_NONE,
        None,
    )

    result = result.drop(
        columns=[
            "_enhance_s",
            "_layout_s",
            "_binarize_s",
            "_vlm_s",
        ],
    )

    return result


# =========================================================
# CLI
# =========================================================

def parse_args():

    parser = argparse.ArgumentParser(
        description="Evaluate OCR/VLM extraction results."
    )

    parser.add_argument(
        "--results",
        required=True,
        help="Prediction results parquet.",
    )

    parser.add_argument(
        "--gold",
        required=True,
        help="Gold-standard CSV.",
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output article-level evaluation CSV.",
    )

    parser.add_argument(
        "--enhance-parquet",
        required=True,
        help="Parquet containing enhancement timing.",
    )

    parser.add_argument(
        "--binarize-parquet",
        required=True,
        help="Parquet containing binarization timing.",
    )

    parser.add_argument(
        "--layout-parquet",
        default=None,
        help="Optional parquet containing layout-detection timing.",
    )

    return parser.parse_args()


# =========================================================
# Main
# page-level:
# TP = correctly matched article
# FP = prediction with no gold article
# FN = gold article with no prediction
# =========================================================

def main():

    args = parse_args()

    results_df = pd.read_parquet(
        args.results
    )

    gold_df = pd.read_csv(
        args.gold,
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        encoding="utf-8",
    )

    enhance_df = pd.read_parquet(
        args.enhance_parquet
    )

    binarize_df = pd.read_parquet(
        args.binarize_parquet
    )

    layout_df = (
        pd.read_parquet(args.layout_parquet)
        if args.layout_parquet
        else None
    )

    article_df, page_df = evaluate(
        results_df,
        gold_df,
    )

    article_df = add_timing(
        article_df,
        enhance_df,
        binarize_df,
        layout_df,
    )

    # -----------------------------------------------------
    # Article-level output
    # -----------------------------------------------------

    article_df.to_csv(
        args.output_csv,
        index=False,
    )

    # -----------------------------------------------------
    # Page-level output
    #
    # This is intentionally separate from the article CSV.
    # -----------------------------------------------------

    page_output = Path(
        args.output_csv
    ).with_name(
        "page_metrics.csv"
    )

    page_df.to_csv(
        page_output,
        index=False,
    )

    print(
        f"Wrote article metrics: {args.output_csv}"
    )

    print(
        f"Wrote page metrics:    {page_output}"
    )


if __name__ == "__main__":
    main()