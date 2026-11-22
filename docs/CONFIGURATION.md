# Configuration

Configuration keeps MA-UP portable across workstations and compute servers without hard-coding machine-specific paths or credentials.

## Principles

- Never commit secrets or API tokens.
- Prefer project-root-relative paths (`data/...`, `scripts/...`) and CLI flags over implicit working-directory assumptions.
- Record explicit `config_id`s, model identifiers, and prompt filenames for every run.
- Keep generated datasets, OCR parquets, and model outputs outside Git tracking.

## Configuration categories

### Paths

Canonical project paths:

- **Sample manifest:** `scripts/R/get_sample/sample.csv`
- **OCR outputs:** `data/corpus_construction/ocr_extraction/<newspaper>/{cropped,none}/ocr.parquet`
- **Gold annotations:** `data/corpus_construction/evaluate_extraction/<newspaper>.csv`
- **LLM extraction outputs:** `data/corpus_construction/llm_extraction/<newspaper>/`
- **DSPy cached datasets:** `data/corpus_construction/local_dspy_llm/datasets/<newspaper>.parquet`
- **DSPy compiled artifacts:** `data/corpus_construction/local_dspy_llm/artifacts/<newspaper>.json`
- **Standalone prompts:** `data/corpus_construction/local_dspy_llm/prompts/{base_prompt.md, <newspaper>_prompt.md}`
- **Cross-validation reports:** `data/corpus_construction/local_dspy_llm/reports/<newspaper>_cv.csv`

### Credentials

Local inference runs against a local Ollama server (`http://127.0.0.1:11434`) using a placeholder key (`api_key="ollama"`). If external endpoints or portals are used during acquisition, supply credentials via environment variables—never in Python source files, notebooks, or tracked config files.

### Models & runtime parameters

Record both the stage configuration and Ollama runtime settings:

```text
model names      : qwen2.5:7b, mistral:7b, deepseek-r1:7b
provider/runtime : ollama_chat via http://127.0.0.1:11434
temperature      : 0.0
max_tokens       : 16384
num_ctx          : 32768
repeat_penalty   : 1.1 (repeat_last_n = 128)
timeout          : 180s
```

### Selected newspaper configurations

Each newspaper configuration binds a specific preprocessing, layout, OCR, LLM, and prompt setup:

| Newspaper | Position | `config_id` | Detector | Binarization | OCR Engine | LLM Model | Active Prompt |
| :--- | :--- | :---: | :--- | :--- | :--- | :--- | :--- |
| `correo` | Interior | 13 | `histogram` | `basic` | `docling_rapidocr` | `mistral:7b` | `base_prompt.md` |
| `elcomercio` | Interior | 51 | `None` (full-page) | `none_color` | `docling_rapidocr` | `qwen2.5:7b` | `base_prompt.md` |
| `gestion` | Interior | 6 | `None` (full-page) | `otsu` | `docling_rapidocr` | `qwen2.5:7b` | `gestion_prompt.md` |
| `ojo` | Interior | 52 | `ppdoclayout` | `basic` | `docling_rapidocr` | `qwen2.5:7b` | `base_prompt.md` |
| `peru21` | Interior | 38 | `ppdoclayout` | `none_color` | `docling_rapidocr` | `deepseek-r1:7b` | `base_prompt.md` |
| `publimetro` | Interior | 12 | `histogram` | `none_grayscale` | `docling_easyocr` | `mistral:7b` | `publimetro_prompt.md` |
| `trome_last` | Last | 7 | `doclayout` | `otsu` | `docling_rapidocr` | `qwen2.5:7b` | `base_prompt.md` |
| `trome_interior` | Interior | 34 | `ppdoclayout` | `none_color` | `docling_rapidocr` | `mistral:7b` | `base_prompt.md` |

### Compute

- Local 7B models require sufficient RAM/VRAM for `num_ctx=32768` and `max_tokens=16384`.
- Thread concurrency (`--threads`, default `2`–`4`) should be tuned to available GPU/CPU capacity so Ollama requests do not time out.

## Configuration validation

Before running extraction or optimization, workflows verify:

- existence of `sample.csv`, `ocr.parquet`, and `<newspaper>.csv`;
- presence of required filter columns (`config_id`, `detector`, `binarization`, `ocr_extractor`) in `ocr.parquet`;
- reachability of the configured Ollama host (`--host`).