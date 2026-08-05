"""
evaluate_extraction.py

Evaluate OCR/VLM extraction results against a gold-standard CSV.

Gold CSV:

    image_stem,title,body

For each gold article and each

    (preprocessing_config, detector, vlm)

combination,

the script:

1. Restricts candidates to the same image_stem
2. Considers only rows with status == "success"
3. Finds the extracted article with the highest body token F1
4. Computes evaluation metrics against that best match

Output:

One row per

    gold article × preprocessing_config × detector × vlm

Thresholded matching:
- minimum article similarity
- fuzzy thresholds for conditional CER/WER
"""

from rapidfuzz.fuzz import ratio
import argparse
import re
import unicodedata
from pathlib import Path
from typing import List
import csv
from scipy.optimize import linear_sum_assignment
import numpy as np
import pandas as pd

COMBO_COLS = [
    "config_id",
    "detector",
    "binarize_file",
    "vlm",
]
TITLE_FUZZY_THRESHOLD = 75
BODY_FUZZY_THRESHOLD = 75
MIN_MATCH_SCORE = 0.20


# ---------------------------------------------------------
# Fuzzy matching
# ---------------------------------------------------------

def fuzzy_score(hyp, ref):

    if not hyp and not ref:
        return 100

    if not hyp or not ref:
        return 0

    return ratio(
        hyp,
        ref
    )

def conditional_error_metric(
    hyp,
    ref,
    matched,
    metric_fn,
):

    if not matched:
        return None

    return metric_fn(
        hyp,
        ref,
    )

# ---------------------------------------------------------
# Normalization
# ---------------------------------------------------------

def normalize_text(text):
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return ""
    
    text = str(text)

    text = unicodedata.normalize("NFKD", text)

    text = "".join(
        c for c in text
        if not unicodedata.combining(c)
    )

    text = text.lower()

    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


# ---------------------------------------------------------
# Levenshtein
# ---------------------------------------------------------

def _levenshtein(a: List, b: List):

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    prev = list(range(len(b)+1))

    for i, ca in enumerate(a, start=1):

        curr = [i] + [0]*len(b)

        for j, cb in enumerate(b, start=1):

            cost = 0 if ca == cb else 1

            curr[j] = min(
                prev[j] + 1,
                curr[j-1] + 1,
                prev[j-1] + cost,
            )

        prev = curr

    return prev[-1]


def cer(hyp, ref):

    if not ref:
        return 0 if not hyp else 1

    return _levenshtein(
        list(hyp),
        list(ref)
    ) / len(ref)


def wer(hyp, ref):

    ref_words = ref.split()

    hyp_words = hyp.split()

    if not ref_words:
        return 0 if not hyp_words else 1

    return _levenshtein(
        hyp_words,
        ref_words
    ) / len(ref_words)


# ---------------------------------------------------------
# Token F1
# ---------------------------------------------------------

def token_f1(hyp, ref):

    from collections import Counter

    hyp_tokens = hyp.split()

    ref_tokens = ref.split()

    if not hyp_tokens and not ref_tokens:
        return 1,1,1

    if not hyp_tokens or not ref_tokens:
        return 0,0,0

    common = Counter(hyp_tokens) & Counter(ref_tokens)

    overlap = sum(common.values())

    if overlap == 0:
        return 0,0,0

    precision = overlap / len(hyp_tokens)

    recall = overlap / len(ref_tokens)

    f1 = (
        2 * precision * recall
        / (precision + recall)
    )

    return precision, recall, f1

def article_similarity(
    pred_title,
    pred_body,
    gold_title,
    gold_body,
):
    """
    Weighted similarity used to match predicted
    and gold articles.
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
        0.3 * title_f1
        + 0.7 * body_f1
    )

    return score, title_f1, body_f1

# ---------------------------------------------------------
# Evaluation
# ---------------------------------------------------------

def evaluate(
    results_df,
    gold_df,
    image_stem_col="image_stem",
    title_col="title",
    body_col="body",
):

    article_rows = []
    page_rows = []

    gold = gold_df.copy()
    gold["_title"] = gold[title_col].map(normalize_text)
    gold["_body"] = gold[body_col].map(normalize_text)

    for combo, combo_df in results_df.groupby(COMBO_COLS):
        for image_stem, image_gold in gold.groupby(image_stem_col):

            candidates = combo_df[
                (combo_df["status"] == "success")
                &
                (combo_df["image_stem"] == image_stem)
            ].copy()

            if len(candidates) == 0:
                continue

            candidates["_title"] = candidates["title"].map(normalize_text)
            candidates["_body"] = candidates["body"].map(normalize_text)

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

            similarity = np.zeros(
                (
                    len(gold_articles),
                    len(pred_articles),
                )
            )

            for gi, (_, gt, gb) in enumerate(gold_articles):

                for pi, (_, prow) in enumerate(pred_articles):

                    similarity[gi, pi], _, _ = article_similarity(
                        prow["_title"],
                        prow["_body"],
                        gt,
                        gb,
                    )

            cost = 1 - similarity

            gold_idx, pred_idx = linear_sum_assignment(cost)

            matched_predictions = set()
            matched_gold = set()

            matched = 0

            for gi, pi in zip(gold_idx, pred_idx):

                score = similarity[gi, pi]

                if score < MIN_MATCH_SCORE:
                    continue

                if score == 0:
                    continue

                matched += 1

                matched_predictions.add(pi)
                matched_gold.add(gi)

                gold_row = image_gold.iloc[gi]

                pred_row = pred_articles[pi][1]

                # matched_predictions.add(pi)

                pred_title = pred_row["_title"]
                pred_body = pred_row["_body"]

                gold_title = gold_row["_title"]
                gold_body = gold_row["_body"]

                title_score = token_f1(
                    pred_title,
                    gold_title,
                )[2]

                body_score = token_f1(
                    pred_body,
                    gold_body,
                )[2]

                title_fuzzy = fuzzy_score(
                    pred_title,
                    gold_title
                )

                title_error_type = (
                    "TP"
                    if title_fuzzy >= TITLE_FUZZY_THRESHOLD
                    else "FP"
                )

                body_fuzzy = fuzzy_score(
                    pred_body,
                    gold_body
                )

                body_error_type = (
                    "TP"
                    if body_fuzzy >= BODY_FUZZY_THRESHOLD
                    else "FP"
                )

                title_fuzzy_match = (
                    title_fuzzy >= TITLE_FUZZY_THRESHOLD
                )

                body_fuzzy_match = (
                    body_fuzzy >= BODY_FUZZY_THRESHOLD
                )

                article_rows.append({
                    "image_stem": image_stem,
                    "config_id": combo[0],
                    "detector": combo[1],
                    "binarize_file": combo[2],
                    "vlm": combo[3],

                    "title_error_type": title_error_type,
                    "body_error_type": body_error_type,

                    "title_fuzzy_score": title_fuzzy,
                    "title_fuzzy_match": title_fuzzy >= TITLE_FUZZY_THRESHOLD,
                    "body_fuzzy_score": body_fuzzy,
                    "body_fuzzy_match": body_fuzzy >= BODY_FUZZY_THRESHOLD,

                    "title_cer":
                        conditional_error_metric(
                            pred_title,
                            gold_title,
                            title_fuzzy_match,
                            cer,
                        ),

                    "title_wer":
                        conditional_error_metric(
                            pred_title,
                            gold_title,
                            title_fuzzy_match,
                            wer,
                        ),

                    "body_cer":
                        conditional_error_metric(
                            pred_body,
                            gold_body,
                            body_fuzzy_match,
                            cer,
                        ),

                    "body_wer":
                        conditional_error_metric(
                            pred_body,
                            gold_body,
                            body_fuzzy_match,
                            wer,
                        ),

                    "body_len_delta_chars": len(pred_body) - len(gold_body),
                    # "article_index": pred_row.get("article_index"),
                    # "matching_score": score,
                    # "title_match_f1": title_score,
                    # "body_match_f1": body_score,
                })

            # ---------------------------------------------------------
            # False negatives: gold articles with no prediction
            # ---------------------------------------------------------

            for gi, (_, gt, gb) in enumerate(gold_articles):

                if gi in matched_gold:
                    continue

                article_rows.append({

                    "image_stem": image_stem,

                    "config_id": combo[0],

                    "detector": combo[1],

                    "binarize_file": combo[2],

                    "vlm": combo[3],
                    "title_error_type": "FN",
                    "body_error_type": "FN",

                    "title_fuzzy_score": 0,

                    "title_fuzzy_match": False,

                    "body_fuzzy_score": 0,

                    "body_fuzzy_match": False,

                    "title_cer": None,

                    "title_wer": None,

                    "body_cer": None,

                    "body_wer": None,

                    "body_len_delta_chars": None,

                    # "article_index": None,

                    # "matching_score": 0,

                    # "title_match_f1": 0,

                    # "body_match_f1": 0,
                })

            # ---------------------------------------------------------
            # False positives: predictions with no gold match
            # ---------------------------------------------------------

            for pi, (_, pred_row) in enumerate(pred_articles):

                if pi in matched_predictions:
                    continue

                article_rows.append({

                    "image_stem": image_stem,

                    "config_id": combo[0],

                    "detector": combo[1],

                    "binarize_file": combo[2],

                    "vlm": combo[3],

                    "title_error_type": "FP",
                    "body_error_type": "FP",

                    "title_fuzzy_score": None,

                    "title_fuzzy_match": False,

                    "body_fuzzy_score": None,

                    "body_fuzzy_match": False,

                    "title_cer": None,

                    "title_wer": None,

                    "body_cer": None,

                    "body_wer": None,

                    "body_len_delta_chars": None,

                    # "article_index":
                    #     pred_row.get("article_index"),

                    # "matching_score": 0,

                    # "title_match_f1": 0,

                    # "body_match_f1": 0,
                })

            n_gold = len(gold_articles)
            n_pred = len(pred_articles)

            tp = matched
            fp = n_pred - tp
            fn = n_gold - tp

            precision = (
                tp / (tp + fp)
                if tp + fp
                else 0
            )

            recall = (
                tp / (tp + fn)
                if tp + fn
                else 0
            )

            page_f1 = (
                2 * precision * recall /
                (precision + recall)
                if precision + recall
                else 0
            )

            page_rows.append({

                "image_stem": image_stem,

                "config_id": combo[0],

                "detector": combo[1],

                "binarize_file": combo[2],

                "vlm": combo[3],

                "gold_articles": n_gold,

                "predicted_articles": n_pred,

                "matched_articles": matched,

                "tp_articles": tp,

                "fp_articles": fp,

                "fn_articles": fn,

                "detection_precision": precision,

                "detection_recall": recall,

                "detection_f1": page_f1,
            })

    return (
        pd.DataFrame(article_rows),
        pd.DataFrame(page_rows),
    )


# ---------------------------------------------------------
# Timing
# ---------------------------------------------------------

def add_timing(
    df,
    enhance_df,
    binarize_df,
    layout_df=None,
):
    """
    Join per-image timing from each pipeline stage onto df and
    compute total_pipeline_seconds.

    Join keys (all also joined on image_stem):
        enhance   : config_id
        layout    : config_id, detector   (optional)
        binarize  : config_id, detector, binarize_file
        vlm       : elapsed_s already in df (results parquet)
    """

    enhance_df = enhance_df.copy()
    enhance_df["image_stem"] = (
        enhance_df["image_path"]
        .apply(lambda p: Path(p).stem)
    )

    # --- enhance ---
    enhance_agg = (
        enhance_df
        .groupby(["image_stem", "config_id"])["processing_time_seconds"]
        .sum()
        .reset_index()
        .rename(columns={"processing_time_seconds": "_enhance_s"})
    )

    df = df.merge(
        enhance_agg,
        on=["image_stem", "config_id"],
        how="left",
    )

    # Normalise detector NaN → sentinel so merges on that column work correctly
    # (pandas does not match NaN == NaN in joins)
    DETECTOR_NONE = "__none__"
    df["detector"] = df["detector"].fillna(DETECTOR_NONE)

    # --- layout detection (optional) ---
    if layout_df is not None:
        layout_df = layout_df.copy()
        layout_df["detector"] = layout_df["detector"].fillna(DETECTOR_NONE)

        layout_agg = (
            layout_df
            .groupby(["image_stem", "config_id", "detector"])["elapsed_s"]
            .sum()
            .reset_index()
            .rename(columns={"elapsed_s": "_layout_s"})
        )

        df = df.merge(
            layout_agg,
            on=["image_stem", "config_id", "detector"],
            how="left",
        )
    else:
        df["_layout_s"] = 0.0

    df["_layout_s"] = df["_layout_s"].fillna(0.0)

    # --- binarization ---
    binarize_df = binarize_df.copy()
    binarize_df["detector"] = binarize_df["detector"].fillna(DETECTOR_NONE)

    binarize_agg = (
        binarize_df
        .groupby(["image_stem", "config_id", "detector", "binarize_file"])["elapsed_s"]
        .sum()
        .reset_index()
        .rename(columns={"elapsed_s": "_binarize_s"})
    )

    df = df.merge(
        binarize_agg,
        on=["image_stem", "config_id", "detector", "binarize_file"],
        how="left",
    )

    # Restore NaN for detector sentinel
    df["detector"] = df["detector"].replace(DETECTOR_NONE, None)

    # --- vlm (already in results parquet) ---
    df["_vlm_s"] = df["elapsed_s"].fillna(0.0) if "elapsed_s" in df.columns else 0.0

    # --- total ---
    df["total_pipeline_seconds"] = (
        df["_enhance_s"].fillna(0.0)
        + df["_layout_s"]
        + df["_binarize_s"].fillna(0.0)
        + df["_vlm_s"]
    )

    df = df.drop(
        columns=["_enhance_s", "_layout_s", "_binarize_s", "_vlm_s"],
    )

    return df


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results", required=True)

    parser.add_argument("--gold", required=True)

    parser.add_argument("--output-csv", required=True)

    parser.add_argument("--enhance-parquet", required=True,
                        help="Parquet with enhance_images timing")

    parser.add_argument("--binarize-parquet", required=True,
                        help="Parquet with binarization timing")

    parser.add_argument("--layout-parquet", default=None,
                        help="Parquet with layout detection timing (optional)")

    return parser.parse_args()


def main():

    args = parse_args()

    results_df = pd.read_parquet(args.results)

    gold_df = pd.read_csv(
        args.gold,
        quotechar='"',
        quoting=csv.QUOTE_ALL,
        encoding="utf-8",
    )

    enhance_df = pd.read_parquet(args.enhance_parquet)
    binarize_df = pd.read_parquet(args.binarize_parquet)
    layout_df = (
        pd.read_parquet(args.layout_parquet)
        if args.layout_parquet
        else None
    )

    article_df, _ = evaluate(
        results_df,
        gold_df,
    )

    article_df = add_timing(
        article_df,
        enhance_df,
        binarize_df,
        layout_df,
    )

    # page_df = add_timing(
    #     page_df,
    #     enhance_df,
    #     binarize_df,
    #     layout_df,
    # )

    article_df.to_csv(
        args.output_csv,
        index=False,
    )

    # page_df.to_csv(
    #     Path(args.output_csv).with_name(
    #         "page_metrics.csv"
    #     ),
    #     index=False,
    # )

    # print_summary(article_df)


if __name__ == "__main__":

    main()