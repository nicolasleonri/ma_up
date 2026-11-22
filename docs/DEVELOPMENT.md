# Development

## Development goals

Changes to MA-UP should preserve end-to-end reproducibility, maintain compatibility with persisted intermediate artifacts, and keep orchestration separate from extraction and evaluation logic.

## Before changing a workflow

Inspect:

1. the entry point (`src/workflows/...` or `src/corpus_construction/.../pipeline.py`);
2. domain logic in `steps.py`;
3. the configuration dictionary (`NEWSPAPERS`) and prompt templates (`prompts/*.md`);
4. the evaluation contract in `src/workflows/evaluate_extraction.py`.

## Keep orchestration thin

Separate CLI parsing and loop orchestration (`pipeline.py`) from domain operations (`steps.py`):

```python
# Preferred pattern
pages = load_dataset(name)
metrics = step_evaluate_pages(program, pages, verbose=True)
```

Avoid duplicating JSON parsing, Hungarian matching, or OCR filtering across multiple scripts. Reuse `similarity_score`, `cer`, `wer`, and `MATCH_THRESHOLD` directly from `src/workflows/evaluate_extraction.py`.

## Prompt & schema conventions

- Standalone prompts live in `data/corpus_construction/local_dspy_llm/prompts/` (`base_prompt.md` and `<newspaper>_prompt.md`) and use a single `{ocr_text}` placeholder.
- Keep `base_prompt.md` synchronized with `ArticleExtraction` and `BASELINE_DEMOS` via `export_base_prompt_markdown()`.
- Preserve verbatim Spanish OCR text in extraction outputs—do not instruct models to autocorrect historical spelling or OCR noise.

## Testing

Tests should verify:

- deterministic page filtering by `config_id`, `detector`, `binarization`, `ocr_extractor`, and `page_position`;
- JSON parsing edge cases (Markdown fences, `<think>...</think>` reasoning blocks, empty arrays `[]`, malformed trailing text);
- Hungarian matching (`evaluate_page_predictions`) on edge cases (0 gold articles, 0 predictions, partial matches around `MATCH_THRESHOLD = 70.0`);
- standalone Markdown prompt export and `{ocr_text}` substitution.

## Logging

Log messages should identify:

- `[stage:newspaper]` prefix (e.g., `[build:gestion]`, `[optimize:publimetro]`);
- page/fold progress (`[idx/total] page_id`);
- exact metrics (`F1`, `Title CER`, `Body WER`) and elapsed time;
- explicit skip/save reasons when comparing against `table1_f1`.

## Code quality

- Use type hints (`from __future__ import annotations`) on public functions.
- Keep random seeds (`seed=42`) and sorting keys (`binarize_file`) explicit and deterministic.
- Never hard-code local machine paths or credentials.
- Update `README.md` and `docs/` whenever pipeline stages, `config_id`s, or benchmark results change.