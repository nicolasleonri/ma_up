# pueue Cheatsheet

Queue manager for hosts without Slurm (e.g. `th-leon`). Daemon plus CLI, no root needed. It replaces `sbatch`, `squeue` and `scancel`.

| Slurm | pueue |
|---|---|
| `sbatch script.sh` | `pueue add -- script.sh` |
| `squeue -u $USER` | `pueue status` |
| `scancel <id>` | `pueue kill <id>` |
| partition | group |
| `--array` | loop with `pueue add` |
| `#SBATCH` limits | not enforced |

## Install (no root)

```bash
mkdir -p ~/.local/bin && cd ~/.local/bin
rm -f pueue pueued
curl -fL -o pueue  https://github.com/Nukesor/pueue/releases/download/v4.0.4/pueue-x86_64-unknown-linux-musl
curl -fL -o pueued https://github.com/Nukesor/pueue/releases/download/v4.0.4/pueued-x86_64-unknown-linux-musl
chmod +x pueue pueued
file pueue pueued          # must say: ELF executable
pueue --version
```

- `-f` makes curl fail on a 404. Without it, curl saves the error text as the file, and running it prints `Not: command not found`.
- Asset names differ between releases. List the current ones:
```bash
curl -s https://api.github.com/repos/Nukesor/pueue/releases/latest | grep browser_download_url | grep -i linux
```

## First start

```bash
pueued -d                  # start the daemon in the background
pueue status
pueue shutdown             # stop the daemon
```

The daemon stops if the container restarts. Run `pueued -d` again. Jobs that were running are marked failed. Rerun them with `pueue restart --failed`.

### Error: "Unable to detect the username for the current user"

Cause: the static binary cannot resolve an LDAP username, so it cannot build the default socket path. Set explicit paths in `~/.config/pueue/pueue.yml`:

```bash
pkill pueued 2>/dev/null
mkdir -p ~/.local/share/pueue ~/.local/run/pueue

sed -i \
 -e 's|^\(\s*\)pueue_directory:.*|\1pueue_directory: /home/users/nleon/.local/share/pueue|' \
 -e 's|^\(\s*\)runtime_directory:.*|\1runtime_directory: /home/users/nleon/.local/run/pueue|' \
 -e 's|^\(\s*\)unix_socket_path:.*|\1unix_socket_path: /home/users/nleon/.local/run/pueue/pueue.socket|' \
 -e 's|^\(\s*\)pid_path:.*|\1pid_path: /home/users/nleon/.local/run/pueue/pueue.pid|' \
 ~/.config/pueue/pueue.yml

pueued -d && sleep 1 && pueue status
```

If the error persists, also export the user name:

```bash
export USER=nleon LOGNAME=nleon
echo 'export USER=nleon LOGNAME=nleon' >> ~/.bashrc
```

## Groups

A group is an independent queue with its own parallel limit. Use groups to separate GPU jobs from CPU jobs.

```bash
pueue group                         # list groups and their limits
pueue group add gpu                 # create a group
pueue parallel 2 -g gpu             # run 2 tasks at a time in the group
pueue parallel 1                    # same for the default group
pueue pause -g gpu                  # pause the group (running tasks keep going)
pueue start -g gpu                  # resume the group
pueue clean -g gpu                  # drop finished tasks of the group
pueue group remove gpu              # delete the group (must be empty)
```

- The default group allows 1 parallel task.
- Tasks added to a group start in order as slots free up.

## Tasks

```bash
pueue add -- <command>                          # default group
pueue add -g gpu -- <command>                   # into a group
pueue add -g gpu -l "label" -- <command>        # label shown in status
pueue add -w <workdir> -- <command>             # working directory
pueue add -a <id> -- <command>                  # start after task <id> succeeds
pueue add --delay 3min -- <command>             # do not start before 3 minutes
pueue add -s -- <command>                       # add stopped (start later)
pueue add -p -- <command>                       # print only the task id
```

- Always put `--` before the command.
- A task runs in a fresh shell with your current environment. Activating a venv beforehand is not required if the command uses the venv interpreter by full path, or if the command is a script that activates it.
- Quote commands that contain `&&`, pipes or redirects:
```bash
pueue add -g gpu -- "cd ~/project/leonrios/ma_up && python3 -m src.workflows.enhance_images > logs/a.log 2>&1"
```

Direct call with the venv interpreter:

```bash
pueue add -g gpu -w ~/project/leonrios/ma_up -- \
  ~/project/leonrios/ma_up/venv/bin/python -m src.workflows.enhance_images \
  --input-dir data/... --output-dir data/...
```

## Create a bash script for tasks

A script keeps environment, variables and logging in one place. Submit it with arguments, like `sbatch script.sh arg`.

### 1. Write the script

`scripts/run_llm_extraction.sh`:

```bash
#!/bin/bash
set -euo pipefail

NEWSPAPER=$1; VARIANT=$2; LLM=$3

cd ~/project/leonrios/ma_up
source .venv/corpus_construction/llm_extraction/bin/activate

export VLLM_USE_FLASHINFER_SAMPLER=0
export HF_HOME=~/cache/huggingface
export GPU_MEM_UTIL=0.4

mkdir -p logs data/corpus_construction/llm_extraction/${NEWSPAPER}
echo "===== ${NEWSPAPER} | ${VARIANT} | ${LLM} ====="

python3 -m src.workflows.llm_extraction \
  --ocr-parquet data/corpus_construction/ocr_extraction/${NEWSPAPER}/${VARIANT}/ocr.parquet \
  --output-parquet data/corpus_construction/llm_extraction/${NEWSPAPER}/results_${VARIANT}_${LLM}.parquet \
  --llms ${LLM}
```

Rules:
- First line `#!/bin/bash`.
- `set -euo pipefail` stops the script at the first error, so the task is marked failed.
- Arguments arrive as `$1`, `$2`, `$3`.
- Use absolute paths or `cd` first. The task may start in another directory.
- Activate the venv inside the script.
- Output of the script is stored by pueue. Read it with `pueue log <id>`. A `> logs/x.log 2>&1` redirect is optional.

### 2. Make it executable

```bash
chmod +x scripts/run_llm_extraction.sh
```

### 3. Test it directly

```bash
./scripts/run_llm_extraction.sh correo cropped deepseek     # Ctrl+C after the first batch
```

### 4. Submit one task

```bash
pueue add -g gpu -l correo_cropped_deepseek -- scripts/run_llm_extraction.sh correo cropped deepseek
```

### 5. Submit many tasks (replaces `--array`)

```bash
for n in correo elcomercio gestion ojo peru21 publimetro trome; do
  pueue add -g gpu -l "${n}_cropped_deepseek" -- scripts/run_llm_extraction.sh $n cropped deepseek
done
pueue add -g gpu -l elcomercio_none_deepseek -- scripts/run_llm_extraction.sh elcomercio none deepseek
```

With several models:

```bash
for n in correo elcomercio; do
  for llm in qwen mistral llama deepseek; do
    pueue add -g gpu -l "${n}_${llm}" -- scripts/run_llm_extraction.sh $n cropped $llm
  done
done
```

### 6. Chain steps

```bash
ID=$(pueue add -p -g gpu -- scripts/run_enhance.sh correo)
pueue add -g gpu -a $ID -- scripts/run_layout.sh correo       # starts after $ID succeeds
```

## Monitor

```bash
pueue status                    # all groups
pueue status -g gpu             # one group
pueue log <id>                  # output of a finished or running task
pueue log <id> -l 50            # last 50 lines
pueue follow <id>               # live output (Ctrl+C leaves the task running)
pueue wait -g gpu               # block until the group is finished
nvidia-smi                      # GPU memory
```

## Kill

```bash
pueue kill <id>                 # kill one running task
pueue kill <id1> <id2> <id3>    # kill several
pueue kill -g gpu               # kill all running tasks in a group
pueue kill --all                # kill every running task
pueue kill -s SIGINT <id>       # send a specific signal (SIGINT lets Python exit cleanly)
```

After a kill:

```bash
pueue status                    # killed tasks show as failed
pueue remove <id>               # remove a task from the list (queued or finished)
pueue restart <id>              # run the task again
pueue restart --failed          # rerun all failed tasks
pueue clean                     # drop finished tasks from the list
pueue reset                     # kill everything and clear all queues
```

- Remove a queued task before it starts: `pueue remove <id>`.
- Stop a whole run without losing the queue: `pueue pause -g gpu`, then `pueue kill -g gpu`.
- Free the GPU afterwards and confirm with `nvidia-smi`.

### Task will not die

`pueue kill` signals the script, but child processes (vLLM engine workers) may survive.

```bash
nvidia-smi                                  # memory still used?
ps -u $USER -o pid,etime,cmd | grep -E "python|vllm|EngineCore"
kill <pid>                                  # normal kill
kill -9 <pid>                               # force
pkill -u $USER -f llm_extraction            # all processes matching a name
```

### Kill a process started without pueue (nohup, background)

```bash
ps -u $USER -o pid,etime,cmd | grep python
kill <pid>
jobs                                        # background jobs of the current shell
kill %1                                     # kill job number 1 of this shell
```

## Rules on th-leon

- GPU #2 is shared with others. Run `nvidia-smi` before queueing GPU jobs.
- One vLLM job at `gpu_memory_utilization` 0.5, or two jobs at 0.4 each (`pueue parallel 2 -g gpu`).
- Stagger two vLLM starts with `--delay 3min` on the second job. Parallel memory profiling can miscalculate free memory.
- Two jobs share the same GPU compute. Check `GPU-Util` in `nvidia-smi` with one job first.
- Free GPU memory when done. Confirm with `nvidia-smi`.
- Make scripts resumable. A container restart kills running tasks.
- Outputs: `~/scratch/leonrios` for temporary files, `~/data/leonrios` for final results.