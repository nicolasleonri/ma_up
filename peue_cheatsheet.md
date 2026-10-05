# pueue Cheatsheet

Queue manager for hosts without Slurm. Daemon plus CLI, no root needed.

## Install (no root)

```bash
mkdir -p ~/.local/bin && cd ~/.local/bin
curl -LO https://github.com/Nukesor/pueue/releases/latest/download/pueue-linux-x86_64
curl -LO https://github.com/Nukesor/pueue/releases/latest/download/pueued-linux-x86_64
mv pueue-linux-x86_64 pueue; mv pueued-linux-x86_64 pueued
chmod +x pueue pueued
pueue --version

cd ~/.local/bin
rm -f pueue pueued
curl -fL -o pueue  https://github.com/Nukesor/pueue/releases/download/v4.0.4/pueue-x86_64-unknown-linux-musl
curl -fL -o pueued https://github.com/Nukesor/pueue/releases/download/v4.0.4/pueued-x86_64-unknown-linux-musl
chmod +x pueue pueued
file pueue pueued
pueue --version

pueued -d
sleep 1
pueue status
pueue add -g gpu -l test -- scripts/run_llm_extraction.sh correo cropped deepseek
pueue follow <id>
pueue kill <id>
for n in elcomercio gestion ojo peru21 publimetro trome; do
  pueue add -g gpu -l "${n}_cropped_deepseek" -- scripts/run_llm_extraction.sh $n cropped deepseek
done
pueue add -g gpu -l elcomercio_none_deepseek -- scripts/run_llm_extraction.sh elcomercio none deepseek
pueue status -g gpu
pueue log <id>
nvidia-smi
```

Verify the filenames on the GitHub release page if the download fails.

## Daemon

```bash
pueued -d            # start in the background
pueue shutdown       # stop the daemon
```

The daemon stops if the container restarts. Run `pueued -d` again afterwards.

## Add jobs

```bash
pueue add -- <command>
pueue add -w <workdir> -- <command>            # set working directory
pueue add -g gpu -- <command>                  # add to a group
pueue add -l "label" -- <command>              # label the job
pueue add -a <id> -- <command>                 # start after job <id> finishes
pueue add -s -- <command>                      # add stopped (do not start)
```

Use the full path to the venv interpreter, no activation needed:

```bash
pueue add -g gpu -w ~/project/leonrios/ma_up -- \
  ~/project/leonrios/ma_up/venv/bin/python -m src.workflows.enhance_images \
  --input-dir data/... --output-dir data/...
```

## Groups and parallelism

```bash
pueue group add gpu
pueue group                     # list groups
pueue parallel 1 -g gpu         # one GPU job at a time
pueue parallel 2                # default group: two at a time
pueue group remove gpu
```

## Monitor

```bash
pueue status
pueue status -g gpu
pueue log <id>                  # output of a finished or running job
pueue follow <id>               # live output (Ctrl+C leaves the job running)
pueue follow                    # follow the oldest running job
```

## Control

```bash
pueue pause <id>
pueue start <id>
pueue kill <id>
pueue restart <id>
pueue restart --failed          # rerun all failed jobs
pueue remove <id>
pueue clean                     # drop finished jobs from the list
pueue reset                     # kill everything and clear the queue
pueue stash <id>                # park a job
pueue enqueue <id>              # bring a stashed job back
```

## Job array substitute

```bash
for i in $(seq 0 3455); do
  pueue add -g gpu -l "cfg_$i" -- ~/project/leonrios/ma_up/venv/bin/python -m src.run --config $i
done
```

Make the script skip configurations with an existing output file, so a rerun resumes.

## Rules on th-leon

- GPU #2 is shared. Run `nvidia-smi` before queueing GPU jobs.
- Set `pueue parallel 1 -g gpu`.
- vLLM: set `gpu_memory_utilization` to 0.5 or lower.
- Free GPU memory when done. Confirm with `nvidia-smi`.
- Outputs: `~/scratch/leonrios` for temporary, `~/data/leonrios` for final results.

## Running jobs without Slurm (pueue)

Hosts without a scheduler (e.g. `th-leon`) use [pueue](https://github.com/Nukesor/pueue) as a queue. It replaces `sbatch`.

### Setup

```bash
pueued -d                    # start the daemon (rerun after a container restart)
pueue group add gpu
pueue parallel 1 -g gpu      # shared GPU: one job at a time
```

### Job script

`scripts/run_enhance.sh`:

```bash
#!/bin/bash
set -euo pipefail
NAME=$1
cd ~/project/leonrios/ma_up
source venv/bin/activate
mkdir -p logs
python3 -m src.workflows.enhance_images \
  --input-dir data/corpus_construction/enhance_images/$NAME \
  --output-dir data/corpus_construction/enhance_images/results/$NAME \
  > logs/enhance_$NAME.log 2>&1
```

### Submit

```bash
pueue add -g gpu -l correo -- scripts/run_enhance.sh correo     # like sbatch

for n in correo comercio gestion; do                            # like --array
  pueue add -g gpu -l $n -- scripts/run_enhance.sh $n
done
```

### Monitor and control

```bash
pueue status
pueue follow <id>
pueue log <id>
pueue kill <id>
pueue restart <id>
pueue add -a <id> -- <command>    # start after job <id> finishes
```

### Notes

- `#SBATCH` resource limits do not exist here. Parallelism comes from `pueue parallel`.
- The GPU is shared. Run `nvidia-smi` before queueing and free GPU memory when done.
- Make scripts resumable. A container restart kills running jobs.