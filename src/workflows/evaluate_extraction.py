"""
evaluate_extraction.py

Evaluate VLM extraction results against a gold-standard CSV.

Gold CSV schema
---------------

    image_stem,title,subheadline,author,body

VLM results schema
------------------

    image_stem
    detector
    config_id
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

Evaluation dimensions
---------------------

Each gold article is evaluated independently for every:

    image_stem
    ×
    config_id
    ×
    detector
    ×
    binarization
    ×
    vlm

The evaluator does NOT select the best extraction among multiple
binarization variants. Every variant is evaluated directly against
the gold standard.

This allows the final experiment to compare:

    enhancement configuration
        ×
    layout detector
        ×
    binarization method
        ×
    VLM

Metrics are calculated independently for:

    - title
    - subheadline
    - author
    - body

For each field:

    - exact match
    - token precision
    - token recall
    - token F1
    - CER
    - WER

Missing or failed VLM extractions are retained in the evaluation
output rather than silently discarded.
"""

import argparse
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import List, Optional

import pandas as pd


# ----------------------------------------------------------------------
# Columns
# ----------------------------------------------------------------------

COMBO_COLS = [
    "config_id",
    "detector",
    "binarization",
    "binarize_file",
    "vlm",
]

GOLD_FIELDS = [
    "title",
    "subheadline",
    "author",
    "body",
]

RESULT_FIELDS = [
    "title",
    "subheadline",
    "author",
    "body",
]


# ----------------------------------------------------------------------
# Normalization
# ----------------------------------------------------------------------

def normalize_text(text) -> str:
    """
    Normalize text before calculating evaluation metrics.

    Steps:

    1. Convert missing values to empty strings.
    2. Apply Unicode NFKD normalization.
    3. Remove combining characters such as accents.
    4. Convert to lowercase.
    5. Normalize whitespace.
    6. Strip leading/trailing whitespace.

    This makes the evaluation less sensitive to differences such as:

        á vs a
        multiple spaces
        line breaks vs spaces
        uppercase vs lowercase
    """

    if text is None:
        return ""

    if isinstance(text, float) and pd.isna(text):
        return ""

    text = str(text)

    text = unicodedata.normalize(
        "NFKD",
        text,
    )

    text = "".join(
        character
        for character in text
        if not unicodedata.combining(character)
    )

    text = text.lower()

    text = re.sub(
        r"\s+",
        " ",
        text,
    )

    return text.strip()


# ----------------------------------------------------------------------
# Levenshtein distance
# ----------------------------------------------------------------------

def _levenshtein(
    a: List,
    b: List,
) -> int:
    """
    Compute Levenshtein edit distance between two sequences.
    """

    if a == b:
        return 0

    if not a:
        return len(b)

    if not b:
        return len(a)

    previous = list(
        range(
            len(b) + 1
        )
    )

    for i, item_a in enumerate(
        a,
        start=1,
    ):

        current = [
            i
        ] + [
            0
        ] * len(b)

        for j, item_b in enumerate(
            b,
            start=1,
        ):

            cost = (
                0
                if item_a == item_b
                else 1
            )

            current[j] = min(
                previous[j] + 1,
                current[j - 1] + 1,
                previous[j - 1] + cost,
            )

        previous = current

    return previous[-1]


# ----------------------------------------------------------------------
# CER
# ----------------------------------------------------------------------

def cer(
    hyp: str,
    ref: str,
) -> float:
    """
    Character Error Rate.

    CER = edit distance / reference length.

    Lower is better.
    """

    if not ref:

        return (
            0.0
            if not hyp
            else 1.0
        )

    return (
        _levenshtein(
            list(hyp),
            list(ref),
        )
        / len(ref)
    )


# ----------------------------------------------------------------------
# WER
# ----------------------------------------------------------------------

def wer(
    hyp: str,
    ref: str,
) -> float:
    """
    Word Error Rate.

    WER = word-level edit distance / reference word count.

    Lower is better.
    """

    ref_words = ref.split()

    hyp_words = hyp.split()

    if not ref_words:

        return (
            0.0
            if not hyp_words
            else 1.0
        )

    return (
        _levenshtein(
            hyp_words,
            ref_words,
        )
        / len(ref_words)
    )


# ----------------------------------------------------------------------
# Token precision / recall / F1
# ----------------------------------------------------------------------

def token_f1(
    hyp: str,
    ref: str,
):
    """
    Calculate token-level precision, recall, and F1.

    Returns
    -------

    precision
    recall
    f1
    """

    hyp_tokens = hyp.split()

    ref_tokens = ref.split()

    # Both empty.
    if (
        not hyp_tokens
        and not ref_tokens
    ):
        return (
            1.0,
            1.0,
            1.0,
        )

    # One empty, one non-empty.
    if (
        not hyp_tokens
        or not ref_tokens
    ):
        return (
            0.0,
            0.0,
            0.0,
        )

    common = (
        Counter(
            hyp_tokens
        )
        &
        Counter(
            ref_tokens
        )
    )

    overlap = sum(
        common.values()
    )

    if overlap == 0:

        return (
            0.0,
            0.0,
            0.0,
        )

    precision = (
        overlap
        / len(hyp_tokens)
    )

    recall = (
        overlap
        / len(ref_tokens)
    )

    f1 = (
        2
        * precision
        * recall
        / (
            precision
            + recall
        )
    )

    return (
        precision,
        recall,
        f1,
    )


# ----------------------------------------------------------------------
# Field-level metrics
# ----------------------------------------------------------------------

def calculate_field_metrics(
    prediction,
    reference,
):
    """
    Calculate all metrics for one extracted field.

    Returns a dictionary containing:

        exact
        precision
        recall
        f1
        cer
        wer
    """

    pred = normalize_text(
        prediction
    )

    gold = normalize_text(
        reference
    )

    precision, recall, f1 = token_f1(
        pred,
        gold,
    )

    return {
        "exact": (
            pred == gold
        ),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": cer(
            pred,
            gold,
        ),
        "wer": wer(
            pred,
            gold,
        ),
    }


# ----------------------------------------------------------------------
# Results validation
# ----------------------------------------------------------------------

def validate_results_columns(
    results_df: pd.DataFrame,
):
    """
    Validate that the VLM results Parquet contains the expected schema.
    """

    required = {
        "image_stem",
        "config_id",
        "detector",
        "binarization",
        "binarize_file",
        "vlm",
        "title",
        "subheadline",
        "author",
        "body",
        "elapsed_s",
        "status",
    }

    missing = (
        required
        - set(
            results_df.columns
        )
    )

    if missing:

        raise ValueError(
            "VLM results Parquet is missing "
            f"required columns: {sorted(missing)}"
        )


# ----------------------------------------------------------------------
# Gold validation
# ----------------------------------------------------------------------

def validate_gold_columns(
    gold_df: pd.DataFrame,
):
    """
    Validate that the gold CSV contains the expected schema.
    """

    required = {
        "image_stem",
        "title",
        "subheadline",
        "author",
        "body",
    }

    missing = (
        required
        - set(
            gold_df.columns
        )
    )

    if missing:

        raise ValueError(
            "Gold CSV is missing "
            f"required columns: {sorted(missing)}"
        )


# ----------------------------------------------------------------------
# Detector normalization
# ----------------------------------------------------------------------

def normalize_detector(
    detector,
) -> Optional[str]:
    """
    Normalize detector values.

    Missing detectors are represented as None internally and as
    the string 'none' in the evaluation output.

    This distinguishes full-page images from cropped images while
    keeping the output Parquet/CSV easy to group and analyze.
    """

    if detector is None:
        return None

    if isinstance(
        detector,
        float,
    ) and pd.isna(
        detector
    ):
        return None

    detector = str(
        detector
    ).strip()

    if (
        not detector
        or detector.lower()
        in {
            "none",
            "nan",
            "null",
        }
    ):
        return None

    return detector


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate(
    results_df: pd.DataFrame,
    gold_df: pd.DataFrame,
):
    """
    Evaluate every VLM extraction against the gold standard.

    For every gold article and every available combination of:

        config_id
        detector
        binarization
        vlm

    the corresponding extraction is evaluated directly.

    No candidate selection or best-F1 matching is performed.

    If an extraction is missing entirely, an evaluation row is still
    produced with:

        extraction_status = "missing"

    If the extraction exists but failed:

        extraction_status = "failed"

    Successful extractions have:

        extraction_status = "ok"
    """

    validate_results_columns(
        results_df
    )

    validate_gold_columns(
        gold_df
    )

    results = results_df.copy()

    gold = gold_df.copy()

    # Normalize detector representation.
    results["detector"] = (
        results["detector"].map(
            normalize_detector
        )
    )

    # Ensure config IDs are consistently represented as integers.
    results["config_id"] = (
        pd.to_numeric(
            results["config_id"],
            errors="coerce",
        )
    )

    # Drop rows where config_id cannot be interpreted.
    results = results[
        results["config_id"].notna()
    ].copy()

    results["config_id"] = (
        results["config_id"].astype(int)
    )

    # Normalize image stems to strings.
    results["image_stem"] = (
        results["image_stem"].astype(str)
    )

    gold["image_stem"] = (
        gold["image_stem"].astype(str)
    )

    # --------------------------------------------------------------
    # Build the unique experimental combinations.
    # --------------------------------------------------------------

    combinations = (
        results[
            COMBO_COLS
        ]
        .drop_duplicates()
        .copy()
    )

    rows = []

    # --------------------------------------------------------------
    # Evaluate each gold article.
    # --------------------------------------------------------------

    for _, gold_row in gold.iterrows():

        image_stem = (
            gold_row[
                "image_stem"
            ]
        )

        # ----------------------------------------------------------
        # Evaluate every available combination.
        # ----------------------------------------------------------

        for _, combo in combinations.iterrows():

            config_id = int(
                combo[
                    "config_id"
                ]
            )

            detector = normalize_detector(
                combo[
                    "detector"
                ]
            )

            binarization = str(
                combo[
                    "binarization"
                ]
            )

            vlm = str(
                combo[
                    "vlm"
                ]
            )

            # ------------------------------------------------------
            # Find the corresponding extraction.
            # ------------------------------------------------------

            candidates = results[
                (results["image_stem"] == image_stem)
                & (results["config_id"] == config_id)
                & (results["binarization"] == binarization)
                & (results["binarize_file"] == binarize_file)
                & (results["vlm"] == vlm)
            ].copy()

            # Detector requires special handling because None/NaN
            # comparisons do not behave like normal string equality.
            if detector is None:

                candidates = candidates[
                    candidates[
                        "detector"
                    ].isna()
                ]

            else:

                candidates = candidates[
                    candidates[
                        "detector"
                    ]
                    == detector
                ]

            # ------------------------------------------------------
            # Select extraction row.
            #
            # There should normally be exactly one row for each
            # unique extraction key.
            #
            # If duplicates exist, keep the latest row.
            # ------------------------------------------------------

            if len(
                candidates
            ) > 0:

                candidate = (
                    candidates.iloc[-1]
                )

                extraction_status = str(
                    candidate.get(
                        "status",
                        "unknown",
                    )
                )

                extraction_error = (
                    candidate.get(
                        "error",
                        None,
                    )
                )

                elapsed_s = (
                    candidate.get(
                        "elapsed_s",
                        None,
                    )
                )

            else:

                candidate = None

                extraction_status = (
                    "missing"
                )

                extraction_error = (
                    "No extraction result found"
                )

                elapsed_s = None

            # ------------------------------------------------------
            # Prediction fields.
            # ------------------------------------------------------

            if candidate is not None:

                pred_title = candidate.get(
                    "title",
                    "",
                )

                pred_subheadline = (
                    candidate.get(
                        "subheadline",
                        "",
                    )
                )

                pred_author = candidate.get(
                    "author",
                    "",
                )

                pred_body = candidate.get(
                    "body",
                    "",
                )

            else:

                pred_title = ""
                pred_subheadline = ""
                pred_author = ""
                pred_body = ""

            # ------------------------------------------------------
            # Calculate metrics.
            #
            # Missing and failed extractions receive zero scores
            # unless the gold field itself is empty.
            # ------------------------------------------------------

            title_metrics = (
                calculate_field_metrics(
                    pred_title,
                    gold_row[
                        "title"
                    ],
                )
            )

            subheadline_metrics = (
                calculate_field_metrics(
                    pred_subheadline,
                    gold_row[
                        "subheadline"
                    ],
                )
            )

            author_metrics = (
                calculate_field_metrics(
                    pred_author,
                    gold_row[
                        "author"
                    ],
                )
            )

            body_metrics = (
                calculate_field_metrics(
                    pred_body,
                    gold_row[
                        "body"
                    ],
                )
            )

            # ------------------------------------------------------
            # Store evaluation result.
            # ------------------------------------------------------

            rows.append(
                {
                    # ------------------------------------------------
                    # Experiment identifiers
                    # ------------------------------------------------

                    "image_stem": image_stem,

                    "config_id": config_id,

                    "detector": (
                        detector
                        if detector is not None
                        else "none"
                    ),

                    "binarization": (
                        binarization
                    ),

                    "binarize_file": (
                        combo.get(
                            "binarize_file",
                            None,
                        )
                    ),

                    "vlm": vlm,

                    # ------------------------------------------------
                    # Extraction metadata
                    # ------------------------------------------------

                    "extraction_status": (
                        extraction_status
                    ),

                    "extraction_error": (
                        extraction_error
                    ),

                    "elapsed_s": (
                        elapsed_s
                    ),

                    # ------------------------------------------------
                    # Title
                    # ------------------------------------------------

                    "gold_title": (
                        gold_row[
                            "title"
                        ]
                    ),

                    "pred_title": (
                        pred_title
                    ),

                    "title_exact": (
                        title_metrics[
                            "exact"
                        ]
                    ),

                    "title_precision": (
                        title_metrics[
                            "precision"
                        ]
                    ),

                    "title_recall": (
                        title_metrics[
                            "recall"
                        ]
                    ),

                    "title_f1": (
                        title_metrics[
                            "f1"
                        ]
                    ),

                    "title_cer": (
                        title_metrics[
                            "cer"
                        ]
                    ),

                    "title_wer": (
                        title_metrics[
                            "wer"
                        ]
                    ),

                    # ------------------------------------------------
                    # Subheadline
                    # ------------------------------------------------

                    "gold_subheadline": (
                        gold_row[
                            "subheadline"
                        ]
                    ),

                    "pred_subheadline": (
                        pred_subheadline
                    ),

                    "subheadline_exact": (
                        subheadline_metrics[
                            "exact"
                        ]
                    ),

                    "subheadline_precision": (
                        subheadline_metrics[
                            "precision"
                        ]
                    ),

                    "subheadline_recall": (
                        subheadline_metrics[
                            "recall"
                        ]
                    ),

                    "subheadline_f1": (
                        subheadline_metrics[
                            "f1"
                        ]
                    ),

                    "subheadline_cer": (
                        subheadline_metrics[
                            "cer"
                        ]
                    ),

                    "subheadline_wer": (
                        subheadline_metrics[
                            "wer"
                        ]
                    ),

                    # ------------------------------------------------
                    # Author
                    # ------------------------------------------------

                    "gold_author": (
                        gold_row[
                            "author"
                        ]
                    ),

                    "pred_author": (
                        pred_author
                    ),

                    "author_exact": (
                        author_metrics[
                            "exact"
                        ]
                    ),

                    "author_precision": (
                        author_metrics[
                            "precision"
                        ]
                    ),

                    "author_recall": (
                        author_metrics[
                            "recall"
                        ]
                    ),

                    "author_f1": (
                        author_metrics[
                            "f1"
                        ]
                    ),

                    "author_cer": (
                        author_metrics[
                            "cer"
                        ]
                    ),

                    "author_wer": (
                        author_metrics[
                            "wer"
                        ]
                    ),

                    # ------------------------------------------------
                    # Body
                    # ------------------------------------------------

                    "gold_body": (
                        gold_row[
                            "body"
                        ]
                    ),

                    "pred_body": (
                        pred_body
                    ),

                    "body_exact": (
                        body_metrics[
                            "exact"
                        ]
                    ),

                    "body_precision": (
                        body_metrics[
                            "precision"
                        ]
                    ),

                    "body_recall": (
                        body_metrics[
                            "recall"
                        ]
                    ),

                    "body_f1": (
                        body_metrics[
                            "f1"
                        ]
                    ),

                    "body_cer": (
                        body_metrics[
                            "cer"
                        ]
                    ),

                    "body_wer": (
                        body_metrics[
                            "wer"
                        ]
                    ),

                    "body_len_delta_chars": (
                        len(
                            normalize_text(
                                pred_body
                            )
                        )
                        -
                        len(
                            normalize_text(
                                gold_row[
                                    "body"
                                ]
                            )
                        )
                    ),
                }
            )

    output = pd.DataFrame(
        rows
    )

    # --------------------------------------------------------------
    # Stable output ordering.
    # --------------------------------------------------------------

    if not output.empty:

        output = (
            output.sort_values(
                [
                    "image_stem",
                    "config_id",
                    "detector",
                    "binarization",
                    "vlm",
                ]
            )
            .reset_index(
                drop=True
            )
        )

    return output


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def parse_args():

    parser = argparse.ArgumentParser(
        description=(
            "Evaluate VLM extraction results "
            "against a gold-standard CSV."
        )
    )

    parser.add_argument(
        "--results",
        required=True,
        help=(
            "Path to VLM extraction results "
            "Parquet."
        ),
    )

    parser.add_argument(
        "--gold",
        required=True,
        help=(
            "Path to gold-standard CSV."
        ),
    )

    parser.add_argument(
        "--output-csv",
        required=True,
        help=(
            "Path for the evaluation CSV."
        ),
    )

    return parser.parse_args()


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():

    args = parse_args()

    results_df = pd.read_parquet(
        args.results
    )

    gold_df = pd.read_csv(
        args.gold
    )

    evaluation_df = evaluate(
        results_df,
        gold_df,
    )

    output_path = Path(
        args.output_csv
    )

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    evaluation_df.to_csv(
        output_path,
        index=False,
    )

    print(
        f"Evaluation complete."
    )

    print(
        f"Rows written: "
        f"{len(evaluation_df)}"
    )

    print(
        f"Output: "
        f"{output_path}"
    )

    if not evaluation_df.empty:

        print(
            "\nPreview:"
        )

        print(
            evaluation_df.head()
        )


if __name__ == "__main__":

    main()
