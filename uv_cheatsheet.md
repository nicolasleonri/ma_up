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

uv keeps downloaded wheels and built packages in a cache. Nothing in it is required by an existing environment, so all of it is safe to delete.

### Inspect

```bash
# First run
cd ~
du -sh .[!.]* 2>/dev/null | sort -h | tail -20
du -h --max-depth=2 ~/.cache 2>/dev/null | sort -h | tail -20
du -h --max-depth=2 ~/.local 2>/dev/null | sort -h | tail -10

uv cache dir                          # cache location (default ~/.cache/uv)
du -sh $(uv cache dir)                # total size
du -h --max-depth=2 $(uv cache dir) | sort -h | tail -20
```

Large folders in `archive-v0/<hash>` are unpacked wheels (torch, `nvidia-*` CUDA libraries, vLLM, paddle). The hash names do not tell you which package is inside.

### Clean

```bash
uv cache prune             # remove unused entries only (low risk)
uv cache prune --ci        # also drop pre-built wheels, keep source-built ones
uv cache clean             # remove everything
uv cache clean <package>   # remove one package, e.g. torch
```

Rules:
- Use the `uv cache` commands. Do not delete `archive-v0/<hash>` folders by hand.
- Cost of cleaning: the next install downloads the packages again.
- Existing environments keep working after `uv cache clean`.
- Space returns only when the environments that hardlink the same files are deleted too.

### Move or bypass

```bash
export UV_CACHE_DIR=~/cache/uv                  # move the cache
echo 'export UV_CACHE_DIR=~/cache/uv' >> ~/.bashrc
uv pip install pkg --no-cache                   # one install without the cache
uv pip install pkg --refresh                    # revalidate cached metadata
```

### Hardlink warning

```
warning: Failed to hardlink files; falling back to full copy.
```

The cache and the environment are on different filesystems (for example cache on `/home`, venv on `/project`). uv copies the files instead. It works but uses double the space and is slower.

```bash
export UV_LINK_MODE=copy      # silence the warning
```

Fix: keep the cache and the venv on the same filesystem, or accept the copy.

### Other caches that grow (not managed by uv)

| Path | Content | Redirect |
|---|---|---|
| `~/.cache/huggingface` | HF models, datasets | `HF_HOME=~/cache/huggingface` |
| `~/.cache/torch` | torch hub weights | `TORCH_HOME=~/cache/torch` |
| `~/.cache/pip` | pip cache | `PIP_CACHE_DIR=~/cache/pip` |
| `~/.cache/flashinfer` | compiled FlashInfer kernels | `XDG_CACHE_HOME=~/cache` |
| `~/.paddlex`, `~/.paddleocr` | PaddleX, PaddleOCR models | none, delete or leave |
| `~/.cache/datalab` | Surya models | `XDG_CACHE_HOME=~/cache` |
| `~/.EasyOCR` | EasyOCR models | none, delete or leave |
| `~/.local/share/uv/python` | uv-managed Python builds | `UV_PYTHON_INSTALL_DIR` |

Check all sizes:

```bash
du -sh ~/.cache/* ~/.paddlex ~/.paddleocr ~/.EasyOCR ~/.local/share/uv 2>/dev/null | sort -h
```

A deleted model cache downloads again on the next run (for example 8 GiB for the DeepSeek 7B FP8 model).

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