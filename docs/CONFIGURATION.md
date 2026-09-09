# Configuration

Configuration should make a workflow portable between a laptop, workstation, and compute cluster without changing source code.

## Principles

- Never commit secrets.
- Prefer environment variables for credentials.
- Prefer explicit paths over implicit working-directory assumptions.
- Record model identifiers and revisions.
- Keep generated data outside the source tree where practical.

## Configuration categories

### Paths

Document paths for:

- raw corpus;
- processed images;
- layout outputs;
- extraction outputs;
- training data;
- model checkpoints;
- evaluation reports.

Use a single configuration mechanism rather than defining paths independently in every workflow.

### Credentials

Credentials for portals, model providers, or external services should be supplied through environment variables or the project's secret-management mechanism.

Never place API keys or passwords directly in:

- Python files;
- notebooks;
- README examples;
- shell history committed to the repository;
- JSON/YAML configuration tracked by Git.

### Models

Record at least:

```text
model name
model revision/version
provider or local runtime
inference parameters
```

For fine-tuning, additionally record the base checkpoint and training configuration.

### Compute

Document hardware assumptions where relevant:

- CPU/GPU requirements;
- GPU memory;
- CUDA/runtime version;
- multiprocessing settings;
- batch size;
- precision.

A workflow should fail with a clear error when required hardware or runtime support is unavailable.

## Reproducible configuration

For an experiment, save the effective configuration alongside the outputs:

```text
run/
├── config.json
├── logs/
├── inputs/
├── outputs/
└── metrics/
```

The exact directory names may be adapted to the project's existing artifact layout.

## Configuration validation

Validate required settings before expensive processing starts.

Examples:

- required input exists;
- output directory is writable;
- model identifier is present;
- credentials are available when required;
- GPU/runtime is available when required;
- incompatible options are rejected.

Fail early rather than discovering configuration errors after hours of processing.
