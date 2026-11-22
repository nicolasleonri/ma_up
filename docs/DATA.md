# Data and Artifacts

## Data lifecycle

MA-UP treats data as a sequence of persisted, inspectable stage artifacts:

```text
source scans + sample.csv
  → prepared / binarized images
  → detected regions + ocr.parquet
  → aligned page datasets (datasets/<newspaper>.parquet)
  → standalone prompts (prompts/*.md) + extracted JSON/Parquet
  → evaluated metrics (reports/<newspaper>_cv.csv)
```

Keep original newspaper scans and gold-standard CSVs immutable and separate from derived artifacts.

## Provenance

Every page and extracted record is tracked via stable identifiers:

- `image_stem` (`page_id`): `<newspaper>_<YYYY-MM-DD>_<page_number>` (e.g., `gestion_1998-05-12_04`);
- `page_position`: `Interior` or `Last` (mapped from `back` in `sample.csv`);
- `config_id`: integer identifier linking detector, binarization, and OCR engine parameters;
- `binarize_file`: crop filename used to preserve deterministic top-to-bottom, left-to-right reading order across regions.

## Core artifact schemas

### 1. Cached page dataset (`datasets/<newspaper>.parquet`)

| Column | Type | Description |
| :--- | :--- | :--- |
| `newspaper` | `str` | Configuration key (e.g., `gestion`, `trome_interior`) |
| `page_id` | `str` | Unique page stem (`image_stem`) |
| `strat` | `str` | Publication year extracted from `page_id` for stratified $k$-fold splitting |
| `n_regions` | `int` | Count of non-empty OCR regions on the page |
| `regions_json` | `str` | JSON list of verbatim OCR strings ordered by `binarize_file` |
| `gold_articles_json` | `str` | JSON list of gold articles (`{"title": ..., "body": ...}`) with non-empty bodies |

### 2. Structured article extraction schema (LLM output)

Each region/page extraction returns a JSON array of objects with exact keys:

```json
[
  {
    "title": "Main headline as it appears (or \"\")",
    "subheadline": "Secondary headline, deck, or standfirst (or \"\")",
    "author": "Byline (or \"\")",
    "body": "Full article body text with \\n paragraph breaks (or \"\")"
  }
]
```

### 3. Cross-validation report (`reports/<newspaper>_cv.csv`)

Records per-fold baseline vs. DSPy-optimized metrics: `newspaper`, `model`, `fold`, `n_test_pages`, `base_f1`, `opt_f1`, `base_title_cer`, `opt_title_cer`, `base_body_wer`, `opt_body_wer`.

## Dataset splits & anti-overfitting policy

- Pages are partitioned using year-stratified $k$-fold cross-validation (`stratified_k_folds`, deterministic `seed=42`).
- Region-level training examples for DSPy (`_align_regions_to_gold`) are constructed strictly from each fold's training pages, including both positive article regions (`fuzz.partial_ratio >= 45.0`) and negative noise regions (ads/captions mapped to `[]`).
- Out-of-fold evaluation runs on held-out test pages. A compiled prompt (`<newspaper>_prompt.md`) is exported and adopted for production only if its out-of-fold `F1` strictly exceeds the Table 1 baseline (`table1_f1`); otherwise, extraction defaults to `base_prompt.md`.

## Data quality checks

- **Pre-extraction:** Drop empty OCR strings while preserving verbatim spelling, punctuation, and OCR artifacts; sort crops deterministically by `binarize_file`; exclude unannotated `NA` gold bodies.
- **Post-extraction:** Strip `<think>...</think>` blocks and Markdown fences; validate JSON list structure; filter empty objects lacking both `title` and `body`.