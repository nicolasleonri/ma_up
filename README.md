# MA-UP

MA-UP is a workflow-oriented research pipeline for processing, extracting, and evaluating structured journalistic articles from historical Peruvian newspaper scans (`Correo`, `El Comercio`, `Gestión`, `Ojo`, `Perú21`, `Publimetro`, `Trome`). The project brings together corpus acquisition, image preparation, layout/region detection, OCR extraction, DSPy prompt optimization, structured LLM extraction, and Hungarian-matching evaluation.

The repository is organized around reusable Python workflow entry points in `src/workflows` and corpus-construction modules in `src/corpus_construction`. The documentation below describes the pipeline stages, selected configurations, and persisted artifacts.

## Pipeline

```text
Source corpus (Newspaper scans + scripts/R/get_sample/sample.csv)
    │
    ▼
Corpus acquisition
    │
    ▼
Image preprocessing (Binarization: basic, otsu, none_color, none_grayscale)
    │
    ▼
Layout / region detection (PPDocLayout, Histogram, YOLO/DocLayout, or Full-Page)
    │
    ▼
OCR extraction (Docling RapidOCR / EasyOCR -> ocr.parquet)
    │
    ▼
Prompt optimization (src/corpus_construction/local_dspy_llm -> winning *_prompt.md)
    │
    ▼
Structured LLM extraction (src/corpus_construction/llm_extraction -> Ollama 7B models)
    │
    ├──────────────► Structured JSON / Parquet outputs
    │
    ▼
Evaluation (src/workflows/evaluate_extraction.py -> Hungarian F1, Title CER, Body WER)
    │
    ▼
Reports / metrics / artifacts
```

The stages are designed to be composable: intermediate artifacts are persisted so that an expensive upstream stage does not have to be repeated when experimenting with a downstream stage.

## Repository layout

```text
.
├── src/
│   ├── workflows/                  # Executable workflow entry points (e.g., evaluate_extraction.py)
│   └── corpus_construction/        # Extraction & optimization modules
│       ├── ocr_extraction/         # Stage 1–4 layout & OCR extraction
│       ├── llm_extraction/         # Production LLM extraction using standalone .md prompts
│       └── local_dspy_llm/         # Stratified k-fold CV DSPy optimization & prompt export
├── scripts/
│   └── R/get_sample/sample.csv     # Sample manifest (newspaper, date, page_number, page_position)
├── data/
│   └── corpus_construction/
│       ├── ocr_extraction/<newspaper>/{cropped,none}/ocr.parquet
│       ├── evaluate_extraction/<newspaper>.csv
│       ├── llm_extraction/<newspaper>/
│       └── local_dspy_llm/
│           ├── datasets/<newspaper>.parquet
│           ├── artifacts/<newspaper>.json
│           ├── prompts/{base_prompt.md, <newspaper>_prompt.md}
│           └── reports/<newspaper>_cv.csv
├── docs/
│   ├── ARCHITECTURE.md
│   ├── WORKFLOWS.md
│   ├── CONFIGURATION.md
│   ├── DATA.md
│   ├── DEVELOPMENT.md
│   └── TROUBLESHOOTING.md
└── README.md
```

> `src/workflows` contains application/research workflows. It is distinct from `.github/workflows`, which is reserved for GitHub Actions CI/CD definitions.

## Main workflow stages

### 1. Corpus acquisition

Retrieves or prepares the source corpus and records page metadata (`newspaper`, `date`, `page_number`, `page_position`) in `scripts/R/get_sample/sample.csv`.

See [docs/WORKFLOWS.md](docs/WORKFLOWS.md).

### 2. Image preprocessing

Raw document images are normalized via binarization filters (`basic`, `otsu`, `none_color`, `none_grayscale`) for subsequent document-analysis stages.

### 3. Layout analysis & OCR extraction

Identifies document regions (`PPDocLayout`, `Histogram`, `YOLO`/`DocLayout`, or full-page `none`) and extracts text via `docling_rapidocr` or `docling_easyocr` into `data/corpus_construction/ocr_extraction/<newspaper>/{cropped,none}/ocr.parquet`.

### 4. Prompt optimization (`src/corpus_construction/local_dspy_llm`)

Refines few-shot demonstrations via DSPy (`BootstrapFewShot`) under stratified $k$-fold cross-validation (`python -m src.corpus_construction.local_dspy_llm.pipeline --build|--eval|--optimize`). Winning out-of-fold prompts that strictly beat baseline F1 are exported as standalone Markdown files (`prompts/<newspaper>_prompt.md`); all other newspapers use `prompts/base_prompt.md`.

### 5. Structured LLM extraction (`src/corpus_construction/llm_extraction`)

Applies local Ollama models (`qwen2.5:7b`, `mistral:7b`, `deepseek-r1:7b`, `num_ctx=32768`, `max_tokens=16384`) with the selected Markdown prompt (`{ocr_text}` placeholder) to extract JSON arrays (`title`, `subheadline`, `author`, `body`).

### 6. Evaluation (`src/workflows/evaluate_extraction.py`)

Compares predictions against gold CSVs (`data/corpus_construction/evaluate_extraction/<newspaper>.csv`) via page-level Hungarian matching (`similarity_score >= 70.0`) and reports **F1**, **Title CER**, and **Body WER**.

## Selected configurations & benchmark results

| Newspaper | Position | Config | Detector | Binarization | OCR | LLM | Prompt | F1 | Title CER | Body WER |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- | :---: | :---: | :---: |
| **Correo** | Interior | 13 | Histogram | `basic` | `rapidocr` | `mistral:7b` | `base_prompt.md` | 0.89 | 0.30 | 0.43 |
| **El Comercio** | Interior | 51 | — | `none_color` | `rapidocr` | `qwen2.5:7b` | `base_prompt.md` | 0.90 | 0.21 | 0.59 |
| **Gestión** | Interior | 6 | — | `otsu` | `rapidocr` | `qwen2.5:7b` | `gestion_prompt.md` | **0.73** | **0.01** | **0.19** |
| **Ojo** | Interior | 52 | PPDocLay | `basic` | `rapidocr` | `qwen2.5:7b` | `base_prompt.md` | 0.77 | 0.85 | 0.01 |
| **Perú21** | Interior | 38 | PPDocLay | `none_color` | `rapidocr` | `deepseek-r1:7b` | `base_prompt.md` | 1.00 | 0.67 | 0.40 |
| **Publimetro** | Interior | 12 | Histogram | `none_grayscale` | `easyocr` | `mistral:7b` | `publimetro_prompt.md` | **0.70** | **0.05** | **0.28** |
| **Trome** | Last | 7 | YOLO/DocLay | `otsu` | `rapidocr` | `qwen2.5:7b` | `base_prompt.md` | 1.00 | 0.31 | 0.12 |
| **Trome** | Interior | 34 | PPDocLay | `none_color` | `rapidocr` | `mistral:7b` | `base_prompt.md` | 0.89 | 0.36 | 0.22 |
| **Mean / Median** | — | — | — | — | — | — | — | **0.86 / 0.89** | **0.35 / 0.31** | **0.28 / 0.25** |

## Getting started

### Requirements

Use the Python version and dependencies declared by the repository's package/environment configuration. Keep the workflow environment consistent across acquisition, preprocessing, inference, and evaluation.

```bash
python -m venv .venv
source .venv/bin/activate
# Install the dependencies specified by the project.
```

On Windows:

```powershell
python -m venv .venv
.venv\Scripts\activate
```

Ensure a local Ollama server is running (`http://127.0.0.1:11434`) with `qwen2.5:7b`, `mistral:7b`, and `deepseek-r1:7b` available.

### Configuration

Do not hard-code credentials, tokens, or machine-specific paths in workflow source files. Keep local configuration in environment variables or ignored configuration files. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### Running workflows

The canonical entry points live in `src/workflows` and `src/corpus_construction`. Run each workflow according to its module's CLI/help interface rather than relying on undocumented positional arguments:

```bash
python -m <workflow_module> --help
```

## Reproducibility

For every experiment, record:

- source corpus/version and `sample.csv` manifest;
- input and output paths;
- preprocessing, detector, binarization, and `config_id` settings;
- OCR engine, LLM model, and Ollama runtime parameters;
- standalone Markdown prompts (`prompts/*.md`) and extraction schema;
- software/environment version and random seeds;
- evaluation configuration and resulting metrics/artifact locations.

Avoid mixing generated data with source code. Keep large datasets, model checkpoints, and generated images outside Git unless the repository explicitly requires them.

## Development principles

The workflows should remain:

1. **Deterministic where possible** — make seeds, `config_id`s, and model versions explicit.
2. **Restartable** — preserve intermediate artifacts.
3. **Observable** — log the stage, inputs, outputs, and failures.
4. **Configurable** — avoid embedding environment-specific values in Python code.
5. **Testable** — keep orchestration separate from domain logic and external services.
6. **Idempotent where practical** — rerunning a completed step should not corrupt existing artifacts.

See [docs/DEVELOPMENT.md](docs/DEVELOPMENT.md) for development guidance.

## Documentation

- [Architecture](docs/ARCHITECTURE.md)
- [Workflows](docs/WORKFLOWS.md)
- [Configuration](docs/CONFIGURATION.md)
- [Data and artifacts](docs/DATA.md)
- [Development](docs/DEVELOPMENT.md)
- [Troubleshooting](docs/TROUBLESHOOTING.md)

## Research / project status

This repository is intended to support an evolving research pipeline. Workflow interfaces and generated artifact formats may change as experiments progress.

When changing a workflow, update the corresponding documentation and record compatibility-breaking changes in the project's change history or release notes.