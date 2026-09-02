# ma_up

Research pipeline for acquiring and constructing a corpus of Peruvian newspaper pages and extracting article text with reproducible OCR/VLM experiments.

> **Repository status:** The current `main` branch implements corpus acquisition, image enhancement, layout detection, binarization, local VLM extraction, and extraction evaluation. The older README described additional annotation, argument-mining, feature-extraction, aggregation, and R-analysis stages that are not currently present in the repository tree; those stages are intentionally not documented here as implemented functionality.

## Overview

`ma_up` is an experimental pipeline for turning digitized newspaper pages into structured text. The current workflow is:

```text
newspaper archive
      │
      ▼
corpus acquisition
      │
      ▼
raw page images + metadata
      │
      ▼
image enhancement
(54 configurations)
      │
      ▼
layout detection
(article crops + metadata)
      │
      ▼
binarization
(8 methods)
      │
      ▼
local VLM extraction
(OLMOCR / DeepSeek / Nanonets)
      │
      ▼
structured article fields
(title / subheadline / author / body)
      │
      ▼
evaluation against gold standard
(CER / WER / token F1)
```

The repository is designed around large experiments and is therefore suitable for HPC/Slurm execution. Most stages support resuming from checkpoints.

## Repository structure

```text
ma_up/
├── requirements/
│   ├── corpus_acquisition.txt
│   ├── enhance_images.txt
│   ├── layout_detection.txt
│   └── vlm_extraction.txt
│
├── scripts/
│
├── src/
│   ├── corpus_acquisition/
│   │   ├── browser.py
│   │   ├── crawl_registry.py
│   │   ├── crawler.py
│   │   ├── downloader.py
│   │   └── metadata_extractor.py
│   │
│   ├── corpus_annotation/
│   │
│   ├── corpus_construction/
│   │   ├── enhance_images/
│   │   ├── layout_detection/
│   │   ├── binarize/
│   │   └── vlm_extraction/
│   │
│   ├── schemas/
│   ├── utils/
│   └── workflows/
│       ├── acquire_corpus.py
│       ├── enhance_images.py
│       ├── layout_detection.py
│       ├── binarize.py
│       ├── vlm_extraction.py
│       ├── evaluate_extraction.py
│       └── finetune_vlm.py
│
├── data/
│   └── corpus_construction/
│
├── logs/
│
├── 0_corpus_acquisition.sh
├── 1_ocr_pipeline_gpu.sh
├── 1_ocr_pipeline_nogpu.sh
├── LICENSE
└── README.md
```

## Installation

The project uses separate requirement files because the stages have very different dependencies. In particular, VLM extraction requires a GPU-oriented environment.

Create a Python environment appropriate for your cluster, then install the requirements for the stage you intend to run:

```bash
python -m venv .venv
source .venv/bin/activate

pip install -r requirements/corpus_acquisition.txt
```

For image enhancement:

```bash
pip install -r requirements/enhance_images.txt
```

For layout detection:

```bash
pip install -r requirements/layout_detection.txt
```

For local VLM extraction:

```bash
pip install -r requirements/vlm_extraction.txt
```

The repository's Slurm scripts are the preferred entry points for the complete HPC workflow.

## Credentials and sensitive configuration

The corpus acquisition workflow accepts a YAML credentials file and defaults to:

```text
config/corpus_acquisition/credentials.yaml
```

Do **not** commit real usernames, passwords, library credentials, or other secrets to Git.

Keep local credentials outside version control and pass the path explicitly:

```bash
python -m src.workflows.acquire_corpus \
  --newspaper elcomercio \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --credentials /path/to/credentials.yaml
```

Before running a public clone of the repository, inspect `.gitignore` and your Git history to ensure credentials have not been committed.

## Corpus acquisition

Available newspaper keys are registered in `src/corpus_acquisition/crawl_registry.py`.

The current registry contains:

- `elcomercio`
- `trome`
- `correo`
- `ojo`
- `peru21`
- `gestion`

Example:

```bash
python -m src.workflows.acquire_corpus \
  --newspaper elcomercio \
  --start-date 2024-01-01 \
  --end-date 2024-01-31 \
  --credentials config/corpus_acquisition/credentials.yaml
```

Useful options:

| Option | Description |
|---|---|
| `--newspaper` | Required newspaper key |
| `--start-date` | First date, `YYYY-MM-DD` |
| `--end-date` | Inclusive final date |
| `--credentials` | Credentials YAML path |
| `--no-resume` | Re-crawl dates already recorded as successful |
| `--headed` | Show the browser UI for debugging |
| `--log-dir` | Directory for run logs |

The workflow writes newspaper metadata under `data/raw/metadata/` and logs under `logs/corpus_acquisition/`.

The acquisition workflow uses browser automation and contains special handling for rate limiting and browser/driver failures.

## Image enhancement

The enhancement stage generates **54 preprocessing configurations**:

- 2 contrast options
- 9 denoising options
- 3 sharpening options

The stages are applied in this order:

```text
Contrast → Denoising → Sharpening
```

Current techniques are:

**Contrast**

- none
- CLAHE

**Denoising**

- none
- mean filter
- Gaussian filter
- median filter
- conservative filter
- Laplacian filter
- frequency filtering
- Crimmins speckle removal
- unsharp filter

**Sharpening**

- none
- unsharp masking
- stroke-width enhancement

Run:

```bash
python -m src.workflows.enhance_images \
  --input-dir data/raw/images \
  --output-dir data/processed/enhanced
```

For one newspaper:

```bash
python -m src.workflows.enhance_images \
  --input-dir data/raw/images \
  --output-dir data/processed/enhanced \
  --newspaper elcomercio
```

Use `--no-resume` to ignore the enhancement checkpoint.

The stage records configuration metadata in `enhance_images.parquet` and writes a `configurations.txt` file describing the 54 configurations.

## Layout detection

Layout analysis finds editorial regions and groups them into article crops. The current workflow CLI exposes:

- `doclayout`
- `ppdoclayout`
- `histogram`

The underlying implementation also contains detector classes for `layoutparser` and `surya`, but they are not enabled by the workflow CLI's current `AVAILABLE_DETECTORS` list.

Example:

```bash
python -m src.workflows.layout_detection \
  --preprocessed-dir data/processed/enhanced/elcomercio \
  --output-dir data/corpus_construction/layout_detection \
  --detectors doclayout ppdoclayout histogram
```

Important options:

```text
--grid-rows
--grid-cols
--score-threshold
--no-resume
```

The default grid is `3 × 3` and the default detection score threshold is `0.5`.

Layout output contains article crops plus metadata describing the source page, preprocessing configuration, detector, article index, bounding box, grid position, number of merged regions, processing time, and status.

The crop naming convention is based on the source page and configuration, for example:

```text
<newspaper>_<date>_<page>_config_<N>_<detector>_article_<M>.tiff
```

## Binarization

Binarization applies the currently implemented set of eight methods:

1. `none_color`
2. `none_grayscale`
3. `basic`
4. `otsu`
5. `adaptive_mean`
6. `adaptive_gaussian`
7. `yannihorne`
8. `niblack`

> The source module contains an outdated docstring saying "7" methods, but the actual method registry contains 8. Documentation follows the executable configuration.

Run:

```bash
python -m src.workflows.binarize \
  --input-dir data/processed/enhanced \
  --output-dir data/corpus_construction/binarization
```

If cropped images are being processed, provide the layout metadata:

```bash
python -m src.workflows.binarize \
  --input-dir data/processed/enhanced \
  --output-dir data/corpus_construction/binarization \
  --layout-parquet data/corpus_construction/layout_detection/results.parquet
```

The stage writes:

```text
<output-dir>/
├── binarization.parquet
├── configurations.txt
└── .binarization_checkpoint.txt
```

The Parquet metadata includes:

- `image_stem`
- `detector`
- `config_id`
- `binarization`
- `binarize_file`
- `elapsed_s`
- `status`

For cropped images, `image_stem`, `detector`, and crop provenance are resolved through the layout Parquet.

## Local VLM extraction

VLM extraction is performed **locally and in-process** with vLLM. No VLM HTTP server is required.

The extraction architecture is:

```text
binarized image
      │
      ▼
local VLM / vLLM
      │
      ▼
raw transcription
      │
      ▼
DSPy structured extraction
      │
      ├── title
      ├── subheadline
      ├── author
      └── body
```

The current selectable VLMs are:

- `olmocr`
- `deepseek`
- `nanonets`

Example:

```bash
python -m src.workflows.vlm_extraction \
  --binarization-parquet \
  data/corpus_construction/binarization/binarization.parquet \
  --binarized-dir \
  data/corpus_construction/binarization \
  --output-parquet \
  data/corpus_construction/vlm_extraction/results.parquet
```

Run one VLM:

```bash
python -m src.workflows.vlm_extraction \
  --binarization-parquet \
  data/corpus_construction/binarization/binarization.parquet \
  --binarized-dir \
  data/corpus_construction/binarization \
  --vlms olmocr
```

Important options:

| Option | Default | Purpose |
|---|---:|---|
| `--vlms` | all | VLMs to run |
| `--max-new-tokens` | 4096 | Generation limit |
| `--batch-size` | 16 | Images per batch |
| `--gpu-memory-utilization` | 0.85 | vLLM GPU memory fraction |
| `--tensor-parallel-size` | 1 | GPUs used to shard a model |
| `--max-model-len` | model default | Optional context-length override |
| `--dtype` | `bfloat16` | Model dtype |
| `--no-skip-failed` | off | Retry previously failed extractions |

The pipeline loads VLMs lazily and is designed to release GPU resources between model runs.

The VLM output Parquet contains:

```text
image_stem
config_id
detector
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
```

### Extraction behavior

The vision prompt is designed for newspaper pages containing multiple columns and multiple independent text blocks. It asks the model to preserve the printed text and reading order without translating, correcting, summarizing, or inventing text.

The structured extraction stage uses DSPy. If DSPy parsing fails, the implementation contains a JSON-parsing fallback.

## Evaluation

`src.workflows.evaluate_extraction` evaluates VLM output against a gold-standard CSV.

The expected gold CSV has:

```csv
image_stem,title,body
```

For each gold article and each combination of:

```text
config_id × detector × VLM
```

the evaluator restricts candidate extractions to the same `image_stem`, considers successful rows, identifies the best matching extracted article, and computes OCR metrics.

The evaluator uses token-level F1 for article matching and reports:

- CER
- WER
- token precision
- token recall
- token F1

The current matching thresholds include:

```text
TITLE_FUZZY_THRESHOLD = 75
BODY_FUZZY_THRESHOLD = 75
MIN_MATCH_SCORE = 0.20
```

The implementation also uses `scipy.optimize.linear_sum_assignment` for assignment between gold and extracted articles.

For the exact command-line interface and gold-standard/output arguments, run:

```bash
python -m src.workflows.evaluate_extraction --help
```

## Resuming and checkpoints

Long-running stages are designed to resume rather than start from scratch.

Current checkpoint files include:

```text
.enhancement_checkpoint.txt
.layout_checkpoint.txt
.binarization_checkpoint.txt
```

VLM extraction uses the output Parquet as its record of completed extraction keys.

Use `--no-resume` only when you intentionally want to recompute an existing stage.

## HPC / Slurm

The repository contains three top-level shell workflows:

```text
0_corpus_acquisition.sh
1_ocr_pipeline_gpu.sh
1_ocr_pipeline_nogpu.sh
```

These scripts should be treated as the canonical examples for cluster execution.

The GPU OCR workflow is appropriate for layout/VLM stages that require GPU resources. The no-GPU workflow is intended for stages that can run without GPU acceleration.

Inspect the shell scripts before submitting because resource requests, environment/module names, paths, and cluster-specific assumptions are infrastructure-specific.

## Data contracts

The pipeline uses Parquet files as the main metadata contracts between stages.

The most important contracts are:

```text
raw metadata
    ↓
enhancement metadata
    ↓
layout metadata
    ↓
binarization metadata
    ↓
VLM extraction results
    ↓
evaluation results
```

Avoid manually changing identifier columns such as `image_stem`, `config_id`, `detector`, and `binarize_file`: downstream stages use these fields to associate files with experiments.

## Troubleshooting

### The acquisition browser fails

Try:

```bash
--headed
```

to inspect the browser interaction. Check the acquisition log under `logs/corpus_acquisition/`.

The workflow uses distinct exit codes for rate limiting and browser/driver failures, which can be useful for Slurm retry logic.

### A stage appears to do nothing

Check whether a checkpoint exists. Most image-processing stages skip inputs recorded as completed.

To force a rerun:

```bash
--no-resume
```

### Binarization cannot resolve cropped images

When processing crops, provide the layout Parquet:

```bash
--layout-parquet <path>
```

The file must contain the layout columns required by the binarization pipeline.

### VLM extraction runs out of GPU memory

Try reducing:

```text
--batch-size
--gpu-memory-utilization
```

or use a larger `--tensor-parallel-size` when the model and cluster support it.

You can also run VLMs separately:

```bash
--vlms olmocr
```

rather than loading all configured models in one workflow.

## Development notes

The codebase is currently experimental. Several modules contain TODOs and compatibility logic for evolving Parquet schemas. When changing an intermediate schema, update both the producer and all consumers.

When adding a new experiment configuration:

1. give it a stable identifier;
2. record the configuration in Parquet;
3. preserve source-image provenance;
4. make the stage resumable when practical;
5. document the new CLI option or data contract.

## What is not currently implemented on `main`

The previous README described the following as completed pipeline stages:

- BERTopic corpus filtering
- manual annotation import
- argument-mining model training/domain adaptation
- semantic/sentiment/POS/discourse feature extraction
- article-level aggregation
- final `unified_corpus.parquet`
- R statistical analysis scripts

Those components are not present in the current `main` source tree, so they are deliberately omitted from the implementation documentation. If they exist in another branch or are planned work, they should be documented separately as planned/future stages rather than as current functionality.
