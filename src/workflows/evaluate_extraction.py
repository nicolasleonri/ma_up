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

No thresholds.
"""

from rapidfuzz.fuzz import ratio
import argparse
import re
import unicodedata
from pathlib import Path
from typing import List
import csv

import pandas as pd

COMBO_COLS = [
    "config_id",
    "detector",
    "binarize_file",
    "vlm",
]
TITLE_FUZZY_THRESHOLD = 75
BODY_FUZZY_THRESHOLD = 75

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

    text = text.lower().strip()

    text = re.sub(r"\s+", " ", text)

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

    rows = []
    gold = gold_df.copy()
    gold["_title"] = gold[title_col].map(normalize_text)
    gold["_body"] = gold[body_col].map(normalize_text)

    for _, gold_row in gold.iterrows():
        image_stem = gold_row[image_stem_col]
        gold_title = gold_row["_title"]
        gold_body = gold_row["_body"]

        for combo, combo_df in results_df.groupby(COMBO_COLS):
            candidates = combo_df[
                (combo_df["status"] == "success")
                &
                (combo_df["image_stem"] == image_stem)
            ].copy()

            if len(candidates) == 0:
                continue

            candidates["_title"] = candidates["title"].map(normalize_text)
            candidates["_body"] = candidates["body"].map(normalize_text)

            best_row = None
            best_f1 = -1

            for _, cand in candidates.iterrows():
                _, _, f1 = token_f1(
                    cand["_body"],
                    gold_body
                )

                if f1 > best_f1:
                    best_f1 = f1
                    best_row = cand

            pred_title = best_row["_title"]
            pred_body = best_row["_body"]

            t_prec, t_rec, t_f1 = token_f1(
                pred_title,
                gold_title,
            )

            b_prec, b_rec, b_f1 = token_f1(
                pred_body,
                gold_body,
            )

            title_fuzzy = fuzzy_score(
                pred_title,
                gold_title
            )

            body_fuzzy = fuzzy_score(
                pred_body,
                gold_body
            )

            title_fuzzy_match = (
                title_fuzzy >= TITLE_FUZZY_THRESHOLD
            )

            body_fuzzy_match = (
                body_fuzzy >= BODY_FUZZY_THRESHOLD
            )

            rows.append({

                "image_stem": image_stem,

                "config_id":
                    combo[0],

                "detector":
                    combo[1],

                "binarize_file":
                    combo[2],

                "vlm":
                    combo[3],

                # "gold_title":
                #     gold_row[title_col],

                # "gold_body":
                #     gold_row[body_col],

                # "pred_title":
                #     best_row["title"],

                # "pred_body":
                #     best_row["body"],

                # "title_exact":
                #     pred_title == gold_title,

                # "body_exact":
                #     pred_body == gold_body,

                "title_fuzzy_score":
                    title_fuzzy,

                "title_fuzzy_match":
                    title_fuzzy >= TITLE_FUZZY_THRESHOLD,

                "body_fuzzy_score":
                    body_fuzzy,

                "body_fuzzy_match":
                    body_fuzzy >= BODY_FUZZY_THRESHOLD,

                # "title_precision":
                #     t_prec,

                # "title_recall":
                #     t_rec,

                # "title_f1":
                #     t_f1,

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

                # "body_precision":
                #     b_prec,

                # "body_recall":
                #     b_rec,

                # "body_f1":
                #     b_f1,

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

                "body_len_delta_chars":
                    len(pred_body)
                    - len(gold_body),
            })

    return pd.DataFrame(rows)


# ---------------------------------------------------------
# CLI
# ---------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument("--results", required=True)

    parser.add_argument("--gold", required=True)

    parser.add_argument("--output-csv", required=True)

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

    out = evaluate(
        results_df,
        gold_df,
    )

    out.to_csv(
        args.output_csv,
        index=False,
    )

    print(out.head())


if __name__ == "__main__":

    main()