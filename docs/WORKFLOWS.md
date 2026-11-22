# Workflows

This document describes the executable workflows under `src/workflows` and stage pipelines under `src/corpus_construction`.

## Workflow responsibilities

### 1. Corpus acquisition

**Purpose:** Prepare the newspaper scan corpus and stratified sampling manifest.

- **Input:** Raw newspaper page images and metadata.
- **Output:** `scripts/R/get_sample/sample.csv` recording `newspaper`, `date`, `page_number`, and `page_position` (`Interior`, `Back`).

### 2. Image preprocessing & layout detection (`src/corpus_construction/ocr_extraction`)

**Purpose:** Normalize scanned images and detect article regions.

- **Binarization options:** `basic`, `otsu`, `none_color`, `none_grayscale`.
- **Layout detectors:** `ppdoclayout`, `histogram`, `doclayout` (YOLO/DocLayout), or full-page (`None`).
- **Output:** Cropped region images (`binarize_file`) sorted in reading order.

### 3. OCR extraction (`src/corpus_construction/ocr_extraction`)

**Purpose:** Transcribe detected crops or full pages into verbatim text.

- **Engines:** `docling_rapidocr`, `docling_easyocr`.
- **Output:** `data/corpus_construction/ocr_extraction/<newspaper>/{cropped,none}/ocr.parquet`.

### 4. DSPy prompt optimization (`src/corpus_construction/local_dspy_llm`)

**Purpose:** Build page-level datasets, evaluate baseline prompts, and optimize few-shot demonstrations via stratified $k$-fold cross-validation.

- **Inputs:** `ocr.parquet`, `evaluate_extraction/<newspaper>.csv`, and `sample.csv`.
- **Outputs:**
  - `data/corpus_construction/local_dspy_llm/datasets/<newspaper>.parquet`
  - `data/corpus_construction/local_dspy_llm/reports/<newspaper>_cv.csv`
  - `data/corpus_construction/local_dspy_llm/prompts/base_prompt.md`
  - `data/corpus_construction/local_dspy_llm/prompts/<newspaper>_prompt.md` (exported only when out-of-fold `F1 > table1_f1`, i.e., `gestion_prompt.md` and `publimetro_prompt.md`)
- **Invocation:**
  ```bash
  python -m src.corpus_construction.local_dspy_llm.pipeline --build
  python -m src.corpus_construction.local_dspy_llm.pipeline --eval
  python -m src.corpus_construction.local_dspy_llm.pipeline --optimize --k 5
  ```

### 5. Structured LLM extraction (`src/corpus_construction/llm_extraction`)

**Purpose:** Extract structured articles (`title`, `subheadline`, `author`, `body`) from OCR regions/pages using local Ollama models (`qwen2.5:7b`, `mistral:7b`, `deepseek-r1:7b`).

- **Inputs:** Filtered `ocr.parquet` rows and standalone Markdown prompt (`<newspaper>_prompt.md` if present, otherwise `base_prompt.md`).
- **Output:** Structured JSON / Parquet extraction results per newspaper.

### 6. Evaluation (`src/workflows/evaluate_extraction.py`)

**Purpose:** Evaluate extracted articles against gold-standard annotations using page-level Hungarian matching (`linear_sum_assignment`).

- **Matching criterion:** Composite fuzzy similarity `0.30 * fuzzy_title + 0.70 * fuzzy_body >= 70.0` (`MATCH_THRESHOLD`).
- **Metrics reported:** Detection **F1** (Precision / Recall), **Title CER** (Character Error Rate on true positives), and **Body WER** (Word Error Rate on true positives).
- **Invocation:**
  ```bash
  python -m src.workflows.evaluate_extraction --help
  ```

## Workflow contract

| Item | Specification |
| :--- | :--- |
| **Inputs** | `sample.csv`, `ocr.parquet`, gold `<newspaper>.csv`, standalone `prompts/*.md` |
| **Outputs** | Cached `.parquet` datasets, `_cv.csv` reports, winning `_prompt.md` files, extraction `.parquet` |
| **Configuration** | `NEWSPAPERS` mapping (`config_id`, `detector`, `binarization`, `ocr_extractor`, `model`, `table1_f1`) |
| **External dependencies** | Local Ollama server (`http://127.0.0.1:11434`) with `qwen2.5:7b`, `mistral:7b`, `deepseek-r1:7b` |
| **Failure behavior** | Logs per-page/per-fold exceptions with timing; skips prompt overwrite if `opt_f1 <= table1_f1` |
| **Resume behavior** | Persists intermediate `.parquet`, `.json`, `.csv`, and `.md` artifacts per stage |

## Recommended execution order

1. Prepare `sample.csv` and raw newspaper scans.
2. Run preprocessing, layout detection, and OCR extraction to produce `ocr.parquet`.
3. Run `--build` to cache aligned page-level datasets (`datasets/<newspaper>.parquet`).
4. Run `--optimize` to execute stratified $k$-fold CV and export `base_prompt.md` plus winning newspaper prompts (`gestion_prompt.md`, `publimetro_prompt.md`).
5. Run structured LLM extraction across the target corpus using the selected prompts and Table 1 `config_id`s.
6. Run `src/workflows/evaluate_extraction.py` to generate final metrics and audit per-page predictions.