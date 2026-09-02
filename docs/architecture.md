# Architecture

## Pipeline

The current implementation is a sequence of file-backed processing stages:

```text
Acquisition
    │
    ├── page images
    └── newspaper metadata
            │
            ▼
Image enhancement
    │
    └── 54 image variants per page
            │
            ▼
Layout detection
    │
    └── article crops + layout metadata
            │
            ▼
Binarization
    │
    └── 8 variants per input image
            │
            ▼
VLM extraction
    │
    └── structured article text
            │
            ▼
Evaluation
    │
    └── OCR quality metrics
```

Each stage has a workflow module under `src/workflows/` and a reusable implementation under `src/corpus_construction/` where appropriate.

## Design principles

### File-backed stage boundaries

Parquet files act as metadata contracts between expensive processing stages. Image files remain on disk while Parquet stores provenance, configuration and status.

### Explicit experiment dimensions

The pipeline preserves experiment dimensions such as:

- preprocessing `config_id`
- layout `detector`
- binarization method
- VLM

This makes it possible to compare combinations rather than treating OCR as a single opaque operation.

### Resume support

Long-running stages use checkpoints or previously written Parquet records. This is important because a complete experiment can involve a large number of image/model combinations.

### Local model inference

VLM inference uses vLLM in-process. There is no dependency on a separate inference server.

## Provenance

The central provenance chain is:

```text
image_stem
  + config_id
  + detector
  + binarization
  + binarize_file
  + vlm
```

These fields identify a particular extraction experiment.

## Detector architecture

The layout implementation contains detector classes and shared utilities for:

1. detecting page regions;
2. assigning regions to composition-grid positions;
3. grouping regions into articles;
4. cropping article images;
5. recording crop provenance.

The workflow currently exposes `doclayout`, `ppdoclayout`, and `histogram`. The implementation also contains `layoutparser` and `surya` detector classes, but they are not currently exposed by the workflow's available-detector list.

## VLM architecture

VLM extraction is explicitly two-phase:

```text
image → raw transcription → structured fields
```

The first phase is visual extraction through vLLM. The second phase uses DSPy to convert the transcription into article objects.

This separation is useful because raw transcription can be retained even if structured parsing needs to be revised later.
