# Workflows

This document describes the application workflows under `src/workflows`.

## Workflow responsibilities

Each workflow should have one clear responsibility and a stable input/output contract.

### Corpus acquisition

**Purpose:** obtain and prepare the source corpus.

**Input:** source portal/service or existing source files.

**Output:** downloaded source artifacts plus acquisition metadata.

Important concerns:

- authentication and credentials;
- rate limiting;
- network failures;
- duplicate detection;
- provenance;
- resumability.

External-site selectors and portal-specific logic should remain isolated from the rest of the pipeline.

### Image preprocessing

**Purpose:** convert source images into analysis-ready artifacts.

Typical operations include:

- format normalization;
- image validation;
- resizing;
- cropping;
- orientation correction;
- quality checks;
- deterministic naming.

Preprocessing should never overwrite irreplaceable source files.

### Layout detection

**Purpose:** identify regions or structural elements in a document.

The output should contain enough metadata to map every detected region back to its source document and image coordinates.

Recommended metadata:

```json
{
  "document_id": "...",
  "image_id": "...",
  "region_id": "...",
  "bbox": [0, 0, 0, 0],
  "label": "...",
  "confidence": 0.0
}
```

Adapt the exact schema to the implementation used by the project.

### VLM extraction

**Purpose:** transform visual document content into structured predictions.

Record:

- model identifier;
- model revision;
- prompt/template version;
- inference parameters;
- source artifact;
- output schema;
- errors and retries.

Structured output should be validated before it is accepted as a successful prediction.

### Fine-tuning

**Purpose:** create a project-specific model from curated examples.

A training run should have an immutable or versioned description of:

- training data;
- validation data;
- model base;
- hyperparameters;
- preprocessing;
- random seed;
- output checkpoint;
- evaluation results.

Avoid relying on an implicit "latest" model when publishing experiment results.

### Evaluation

**Purpose:** measure extraction or detection quality against reference data.

Evaluation should produce machine-readable metrics as well as a human-readable summary.

Where possible, preserve per-example results so that aggregate metrics can be audited.

## Running a workflow

First inspect the workflow's CLI:

```bash
python -m <workflow_module> --help
```

Prefer explicit configuration over hidden defaults:

```bash
python -m <workflow_module> \
  --input <input> \
  --output <output>
```

Use the actual options exposed by the repository rather than copying the example literally.

## Workflow contract

Every workflow should document:

| Item | Required |
|---|---|
| Inputs | Yes |
| Outputs | Yes |
| Configuration | Yes |
| External dependencies | Yes |
| Failure behavior | Yes |
| Resume behavior | Recommended |
| Example invocation | Recommended |

## Recommended execution order

For a complete processing run:

1. Acquire the corpus.
2. Validate source files.
3. Preprocess images.
4. Run layout detection.
5. Run VLM extraction.
6. Produce or update training data if required.
7. Fine-tune a model if required.
8. Evaluate predictions.
9. Archive metrics and run metadata.

Not every experiment needs every stage.
