# Data and Artifacts

## Data lifecycle

The project should treat data as a sequence of immutable or versioned artifacts:

```text
source
  → raw
  → prepared
  → detected
  → extracted
  → evaluated
```

Keep original source material separate from derived data.

## Provenance

Every derived artifact should be traceable to:

- source document/image;
- processing stage;
- processing version;
- model/version where applicable;
- configuration;
- timestamp or run identifier.

Use stable identifiers rather than filenames alone.

## Suggested artifact metadata

```json
{
  "run_id": "...",
  "source_id": "...",
  "stage": "...",
  "created_at": "...",
  "software_version": "...",
  "config": {}
}
```

Adapt this to the actual project schema.

## Large files

Do not commit large generated datasets, model checkpoints, or intermediate image collections unless explicitly required.

Use the project's configured storage mechanism or Git LFS where appropriate.

## Dataset splits

Training and evaluation datasets should have stable, documented splits.

Avoid accidentally evaluating on data that was used for training or tuning.

Record the split definition with each experiment.

## Data quality checks

Before processing, validate:

- file existence;
- file readability;
- supported formats;
- image dimensions;
- duplicate inputs;
- missing metadata.

After processing, validate:

- expected output count;
- output readability;
- schema validity;
- coordinate bounds for detected regions;
- absence of unexpected empty outputs.

## Reproducibility

A result should be reproducible from:

```text
source/version
+
code version
+
environment
+
configuration
+
model/version
```

If any of these changes, treat the resulting experiment as a new run.
