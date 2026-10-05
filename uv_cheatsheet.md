# uv Cheatsheet

## virtualenv / pip → uv (pip-compatible mode)

| Task | virtualenv / pip | uv |
|---|---|---|
| Create env | `python -m venv path` | `uv venv path --python 3.11` |
| Activate | `source path/bin/activate` | same |
| Install | `pip install pkg` | `uv pip install pkg` |
| From file | `pip install -r req.txt` | `uv pip install -r req.txt` |
| Editable | `pip install -e .` | `uv pip install -e .` |
| Uninstall | `pip uninstall pkg` | `uv pip uninstall pkg` |
| List | `pip list` | `uv pip list` |
| Freeze | `pip freeze` | `uv pip freeze` |
| Show | `pip show pkg` | `uv pip show pkg` |
| Upgrade | `pip install -U pkg` | `uv pip install -U pkg` |
| Sync exactly | none | `uv pip sync req.txt` |

## Create an environment in a specific folder

```bash
uv venv /path/to/folder --python 3.11
uv venv venv                          # ./venv (relative to current dir)
uv venv ~/project/leonrios/ma_up/venv
uv venv /path --allow-existing        # target exists and is not empty
```

## Python versions

```bash
uv python list              # available versions
uv python install 3.12      # download a version
uv venv --python 3.12       # use it
```

## Target or run an environment without activating

```bash
uv pip install --python /path/venv/bin/python pkg
/path/venv/bin/python script.py
```

## Lock a dependency set

```bash
uv pip compile requirements.in -o requirements.txt
```

## Project mode (needs `pyproject.toml`)

```bash
uv init              # create a project
uv add pkg           # add a dependency
uv remove pkg
uv sync              # create .venv and install the lockfile
uv run script.py     # run inside the project environment
uv lock
```

## Tools (isolated CLIs)

```bash
uv tool install ruff
uvx ruff check .     # one-off run
```

## Index and wheels

```bash
uv pip install torch --index-url https://download.pytorch.org/whl/cu124
uv pip install pkg --extra-index-url URL
```

## Cache

```bash
uv cache dir
uv cache clean
export UV_CACHE_DIR=~/cache/uv     # move the cache
```

## Maintenance

```bash
uv self update
uv --version
```

## Differences from pip

- `uv pip` fails on conflicts at resolve time, with a clear message.
- uv downloads missing Python versions itself.
- `uv venv` creates no `pip` inside the environment. Use `uv pip`. For plain `pip`: `uv venv --seed`.
- Default environment name in project mode: `.venv`.
- Moving or renaming a venv folder breaks it. Recreate it instead.
- Delete an environment: `rm -rf /path/to/folder`.