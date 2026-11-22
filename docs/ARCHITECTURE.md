# Architecture

## Overview

MA-UP is structured as a staged document-processing pipeline for historical Peruvian newspaper scans. Each stage consumes persisted artifacts from the previous stage and produces deterministic, inspectable outputs that can be reused downstream without repeating expensive upstream compute.

```text
Acquisition (sample.csv + raw scans)
    │
    ▼
Preprocessing (Binarization: basic, otsu, none_color, none_grayscale)
    │
    ▼
Layout analysis (PPDocLayout, Histogram, YOLO/DocLayout, or full-page)
    │
    ▼
OCR extraction (Docling RapidOCR / EasyOCR -> ocr.parquet)
    │
    ▼
DSPy prompt optimization (Stratified k-fold CV -> winning *_prompt.md)
    │
    ▼
Structured LLM extraction (Ollama 7B models + standalone .md prompts)
    │
    ▼
Structured predictions (JSON arrays / Parquet)
    │
    ▼
Evaluation (Page-level Hungarian matching -> F1, Title CER, Body WER)
```

## Design boundaries

### Workflow layer (`src/workflows`)

`src/workflows` is the orchestration layer (e.g., `evaluate_extraction.py`). A workflow is responsible for:

- parsing CLI arguments and validating configuration;
- resolving input/output paths;
- invoking domain modules in `src/corpus_construction`;
- handling stage-level retries and error reporting;
- logging run summaries and evaluation metrics.

### Processing layer (`src/corpus_construction`)

Reusable domain logic is separated by stage under `src/corpus_construction/`:

- `ocr_extraction/`: image binarization, layout region cropping, and OCR transcription.
- `local_dspy_llm/`: dataset alignment (`PageRecord`), stratified $k$-fold cross-validation, `BootstrapFewShot` optimization, and standalone Markdown prompt export (`export_prompt_markdown`).
- `llm_extraction/`: region/page-level structured JSON extraction via local Ollama models using standalone `.md` prompt templates.

### External systems

Local Ollama inference servers (`http://127.0.0.1:11434`), OCR runtimes (`docling_rapidocr`, `docling_easyocr`), and filesystem storage are isolated behind thin adapter functions (`make_lm`, `PageArticleExtractor`) so parsing, Hungarian matching, and dataset construction can be tested independently.

## Artifact-oriented execution

Each stage follows a restartable, artifact-oriented loop:

```text
validate inputs & config_id
    ↓
create output directory
    ↓
process logical unit (page / newspaper configuration)
    ↓
write intermediate artifact (.parquet / .json / .md / .csv)
    ↓
record metrics & elapsed time
    ↓
continue
```

## Configuration

Configuration flows from explicit stage definitions (such as `NEWSPAPERS` in `src/corpus_construction/local_dspy_llm/steps.py`) and CLI arguments rather than scattered environment reads:

```text
configuration (NEWSPAPERS / CLI flags)
     │
     ▼
workflow / pipeline
     │
     ├── acquisition (sample.csv)
     ├── preprocessing & layout (config_id, detector, binarization)
     ├── ocr extraction (ocr_extractor, ocr_variant)
     ├── prompt optimization (model, k-folds, table1_f1 threshold)
     ├── llm extraction (prompt .md, Ollama runtime params)
     └── evaluation (Hungarian matching, MATCH_THRESHOLD = 70.0)
```

## Error handling

Errors preserve full stage context:

- newspaper and `page_id` (`image_stem`);
- `config_id` and stage (`build`, `eval`, `optimize`, `extract`, `evaluate`);
- LLM model identifier and Ollama host;
- exception type and per-page timing.

Malformed JSON or reasoning traces (such as `<think>...</think>` blocks from `deepseek-r1:7b`) are sanitized in `_parse_articles` without silently corrupting structured fields.

## Observability

Each stage reports:

- active newspaper configuration (`config_id`, detector, binarization, OCR engine, LLM model);
- number of pages discovered and pages with non-empty OCR regions;
- per-page and per-fold progress (`F1`, `Title CER`, `Body WER`, elapsed seconds);
- selection decisions (whether a compiled DSPy fold strictly beat the Table 1 baseline);
- output artifact paths.

## Extensibility

Adding a new newspaper, OCR configuration, or LLM model requires adding an entry to the configuration dictionary and ensuring the target Ollama model is pulled—without rewriting orchestration or evaluation logic.