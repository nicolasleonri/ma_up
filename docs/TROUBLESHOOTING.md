# Troubleshooting

## Workflow cannot find an input

Check:

1. the configured input path;
2. the current working directory;
3. whether acquisition completed successfully;
4. whether the expected file extension/format is supported.

Prefer absolute or project-root-relative paths during debugging.

## Authentication or network failures

Check:

- required credentials are available;
- credentials have not expired;
- the external service is reachable;
- rate limits have not been exceeded;
- the workflow is configured for the correct endpoint.

Do not solve authentication failures by committing credentials into the repository.

## Model loading fails

Check:

- model identifier/revision;
- local cache;
- runtime version;
- available disk space;
- available GPU/CPU memory;
- CUDA/runtime compatibility where applicable.

Record the model version used for successful runs.

## Out-of-memory errors

Try, in this order:

1. reduce batch size;
2. reduce image/input resolution if scientifically acceptable;
3. reduce concurrent workers;
4. use an appropriate inference precision;
5. move work to a machine with sufficient memory.

Document any change that can affect model quality.

## Workflow stops part way through

Prefer resuming from persisted intermediate artifacts instead of deleting the output directory and starting again.

If the workflow is not restartable, consider adding:

- per-item checkpoints;
- atomic output writes;
- completed-item manifests;
- retry handling.

## Invalid structured output

Validate model output against a schema before writing it as a successful result.

Keep the original model response or an appropriate diagnostic artifact when possible so failures can be inspected.

## Unexpected evaluation results

Verify:

- dataset split;
- preprocessing version;
- reference annotations;
- model version;
- prompt/schema version;
- evaluation configuration.

A metric change is not necessarily a model change; preprocessing and evaluation changes can also alter results.

## Debugging checklist

```text
[ ] Correct code revision
[ ] Correct environment
[ ] Correct configuration
[ ] Correct input dataset/version
[ ] Correct model/version
[ ] Correct output directory
[ ] Sufficient compute resources
[ ] Relevant logs captured
[ ] Failed artifact preserved
```
