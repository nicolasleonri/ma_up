# Architecture

## Overview

MA-UP is structured as a staged document-processing pipeline. Each stage consumes artifacts from the previous stage and produces artifacts that can be inspected, reused, or passed downstream.

```text
Acquisition
    │
    ▼
Raw corpus
    │
    ▼
Preprocessing
    │
    ▼
Prepared images
    │
    ▼
Layout analysis
    │
    ▼
Regions / layout metadata
    │
    ▼
VLM extraction
    │
    ▼
Structured predictions
    │
    ├──────────────► Evaluation
    │
    └──────────────► Fine-tuning data
                              │
                              ▼
                         Fine-tuned model
                              │
                              ▼
                         Evaluation
```

## Design boundaries

### Workflow layer

`src/workflows` is the orchestration layer. A workflow should be responsible for:

- accepting configuration and CLI arguments;
- determining input/output locations;
- invoking the required processing components;
- handling retries and failures at stage boundaries;
- recording useful logs and metadata.

It should not contain large amounts of reusable domain logic.

### Processing layer

Reusable operations such as image transformations, parsing, schema validation, model invocation, and metric computation should live in dedicated modules rather than being duplicated across workflow scripts.

### External systems

Network services, model providers, portals, storage systems, and GPU-specific runtimes should be isolated behind small interfaces where possible. This makes unit testing possible without requiring access to the external system.

## Artifact-oriented execution

A workflow should preferably follow:

```text
validate inputs
    ↓
create output directory
    ↓
process one logical unit
    ↓
write intermediate artifact
    ↓
record metadata/log
    ↓
continue
```

This allows a failed run to be resumed without restarting the entire pipeline.

## Configuration

Configuration should flow into a workflow from one place. Avoid scattering environment-variable reads throughout business logic.

A recommended separation is:

```text
configuration
     │
     ▼
workflow
     │
     ├── acquisition
     ├── preprocessing
     ├── layout
     ├── extraction
     └── evaluation
```

## Error handling

Errors should retain context:

- workflow/stage;
- input artifact;
- model/service involved;
- operation being performed;
- exception type;
- retry status.

Do not silently catch exceptions that can result in corrupted or incomplete scientific results.

## Observability

At minimum, each workflow should report:

- configuration summary;
- number of inputs discovered;
- number successfully processed;
- number skipped;
- number failed;
- output location;
- elapsed time.

For model inference, also record model identifier and relevant generation/inference settings.

## Extensibility

New models or preprocessing strategies should be added behind stable interfaces rather than by cloning entire workflow scripts.

A new implementation should ideally require changing configuration and dependency wiring rather than rewriting orchestration.
