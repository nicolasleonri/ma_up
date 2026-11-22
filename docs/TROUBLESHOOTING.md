# Troubleshooting

## Missing OCR parquet, gold CSV, or sample manifest

If `step_build_dataset` raises `FileNotFoundError`:

1. Verify you are running commands from the repository root (`MA-UP/`).
2. Confirm `scripts/R/get_sample/sample.csv` exists.
3. Check that `data/corpus_construction/ocr_extraction/<newspaper>/{cropped,none}/ocr.parquet` and `data/corpus_construction/evaluate_extraction/<newspaper>.csv` exist for the target newspaper.
4. For `trome_interior` and `trome_last`, note that both read from `data/corpus_construction/ocr_extraction/trome/cropped/ocr.parquet` and `evaluate_extraction/trome.csv`, filtered by `page_position` (`Interior` vs. `Back`/`Last`).

## Zero OCR regions matched for a newspaper

If `[build:<name>]` reports `0/N with OCR regions`:

1. Inspect `ocr.parquet` columns (`config_id`, `detector`, `binarization`, `ocr_extractor`, `image_stem`).
2. Verify whether the configuration uses `ocr_variant="none"` (`elcomercio`, `gestion`) or `ocr_variant="cropped"` (all others).
3. Ensure `detector` is `None` (or `"none"`) for full-page configs (`51` and `6`).

## Ollama connection errors or timeouts

If `step_evaluate_pages` logs `ERROR` or connection failures:

1. Verify the Ollama daemon is running and reachable at `--host` (default `http://127.0.0.1:11434`).
2. Confirm all required models are pulled locally:
   ```bash
   ollama list
   # Expected: qwen2.5:7b, mistral:7b, deepseek-r1:7b
   ```
3. Lower `--threads` (e.g., `--threads 1` or `--threads 2`) if concurrent requests overwhelm GPU VRAM and trigger the `180s` timeout.

## Truncated JSON or DeepSeek reasoning blocks

- **Truncated JSON on dense pages:** Ensure `max_tokens=16384` and `num_ctx=32768` are set in `make_lm` (and in production `llm_extraction`) so multi-article pages do not clip mid-array.
- **DeepSeek-R1 `<think>` tags:** `deepseek-r1:7b` emits `<think>...</think>` reasoning traces before the JSON array. `_parse_articles` strips these automatically; ensure any custom parser applies the same regex cleanup before `json.loads`.

## DSPy optimization skips saving a prompt

If `[optimize:<name>] SKIPPED save` appears:

- This is expected behavior when no out-of-fold DSPy model strictly outperforms the newspaper's `table1_f1` baseline (`correo`, `elcomercio`, `ojo`, `peru21`, `trome_interior`, `trome_last`).
- Use `data/corpus_construction/local_dspy_llm/prompts/base_prompt.md` for those configurations. Only `gestion` (`0.65 -> 0.73`) and `publimetro` (`0.64 -> 0.70`) export newspaper-specific prompts.

## Debugging checklist

```text
[ ] Executed from repository root
[ ] Virtual environment active with dspy, rapidfuzz, scipy, pandas installed
[ ] Ollama server running with qwen2.5:7b, mistral:7b, deepseek-r1:7b
[ ] sample.csv, ocr.parquet, and gold CSVs present in expected paths
[ ] config_id and ocr_variant match Table 1 specification
[ ] Standalone prompt (.md) contains {ocr_text} placeholder
```