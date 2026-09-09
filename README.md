# MA-UP

MA-UP is a workflow-oriented research pipeline for processing and extracting information from historical/document images. The project brings together corpus acquisition, image preparation, document layout analysis, vision-language extraction, model fine-tuning, and evaluation.

The repository is organized around reusable Python workflow entry points in `src/workflows`. The documentation below describes the intended pipeline, how the stages relate to one another, and how to develop and reproduce runs.

## Pipeline

```text
Source corpus
    │
    ▼
Corpus acquisition
    │
    ▼
Image preprocessing
    │
    ▼
Layout / region detection
    │
    ▼
Vision-language extraction
    │
    ├──────────────► Structured outputs
    │
    ▼
Training / fine-tuning
    │
    ▼
Evaluation
    │
    ▼
Reports / metrics / artifacts
```

The stages are designed to be composable: intermediate artifacts should be persisted so that an expensive upstream stage does not have to be repeated when experimenting with a downstream stage.

## Repository layout

```text
.
├── src/
│   ├── workflows/          # Executable workflow entry points
│   └── ...                 # Supporting application/source code
├── docs/
│   ├── ARCHITECTURE.md
│   ├── WORKFLOWS.md
│   ├── CONFIGURATION.md
│   ├── DATA.md
│   ├── DEVELOPMENT.md
│   └── TROUBLESHOOTING.md
├── README.md
└── ...
```

> `src/workflows` contains application/research workflows. It is distinct from `.github/workflows`, which is reserved for GitHub Actions CI/CD definitions.

## Main workflow stages

### 1. Corpus acquisition

The acquisition stage retrieves or prepares the source corpus and records enough metadata to make the acquisition reproducible.

See [docs/WORKFLOWS.md](docs/WORKFLOWS.md).

### 2. Image preprocessing

Raw document images are normalized for subsequent document-analysis stages. Typical operations include format conversion, resizing, cropping, normalization, and other quality-preserving transformations.

### 3. Layout analysis

The layout stage identifies document regions or structural elements that can be passed to later extraction stages.

### 4. Vision-language extraction

The extraction stage applies a vision-language model to document images or detected regions and produces structured information.

### 5. Fine-tuning

Training workflows prepare examples and launch model fine-tuning when a project-specific model is required.

### 6. Evaluation

Evaluation workflows compare predictions with reference data and produce metrics and/or reports.

## Getting started

### Requirements

Use the Python version and dependencies declared by the repository's package/environment configuration. If multiple environments are supported, keep the workflow environment consistent across acquisition, preprocessing, inference, and evaluation.

Create an isolated environment, then install the project dependencies using the repository's existing package manager configuration.

For example:

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

### Configuration

Do not hard-code credentials, tokens, or machine-specific paths in workflow source files.

Keep local configuration in environment variables or ignored configuration files. See [docs/CONFIGURATION.md](docs/CONFIGURATION.md).

### Running workflows

The canonical entry points live in `src/workflows`. Run each workflow according to its module's CLI/help interface rather than relying on undocumented positional arguments.

A useful discovery pattern is:

```bash
python -m <workflow_module> --help
```

If the repository exposes scripts through its package configuration, prefer those scripts because they provide a stable project-level interface.

## Reproducibility

For every experiment, record:

- source corpus/version;
- input and output paths;
- preprocessing settings;
- model and model revision;
- prompts or extraction schema;
- software/environment version;
- random seeds where applicable;
- evaluation configuration;
- resulting metrics and artifact locations.

Avoid mixing generated data with source code. Keep large datasets, model checkpoints, and generated images outside Git unless the repository explicitly requires them.

## Development principles

The workflows should remain:

1. **Deterministic where possible** — make seeds and model versions explicit.
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
