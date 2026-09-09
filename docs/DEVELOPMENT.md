# Development

## Development goals

Changes to the project should preserve reproducibility while making the workflows easier to test, operate, and extend.

## Before changing a workflow

Read:

1. the workflow entry point;
2. the modules it imports;
3. its input/output contract;
4. the relevant configuration;
5. existing tests;
6. the corresponding documentation.

## Keep orchestration thin

A workflow should coordinate operations rather than implement every operation itself.

Prefer:

```python
def run(config):
    inputs = discover_inputs(config)
    prepared = preprocess(inputs, config)
    predictions = extract(prepared, config)
    return write_outputs(predictions, config)
```

over a single large function containing discovery, image processing, network calls, parsing, and persistence.

## Testing

Tests should cover at least:

- configuration validation;
- input discovery;
- deterministic transformations;
- parsing/schema validation;
- failure handling;
- representative end-to-end execution with external services mocked.

Expensive model inference and network operations should normally be separated from fast unit tests.

## Logging

Use structured, useful logs.

A log message should answer:

- what happened;
- to which artifact;
- during which stage;
- whether the operation succeeded.

Avoid logging secrets or unnecessarily sensitive source data.

## Code quality

Prefer:

- type hints for public functions;
- small functions;
- explicit error handling;
- descriptive names;
- documented non-obvious assumptions;
- deterministic behavior.

Avoid:

- duplicated workflow implementations;
- hard-coded absolute paths;
- hidden global state;
- broad `except Exception` blocks that discard context;
- credentials in source code.

## Pull requests / changes

A workflow change should include:

- implementation;
- tests where practical;
- documentation updates;
- migration notes if outputs change.

If an output schema changes, explicitly document the old and new formats.

## Experiment changes

Research changes should record why the change was made and how it affects comparability with previous runs.

Do not silently change preprocessing or evaluation behavior while keeping the same experiment identifier.
