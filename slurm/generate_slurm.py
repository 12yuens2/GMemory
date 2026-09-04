#!/usr/bin/env python3
"""Generate the per-task slurm/*_experiment.sh, *_crosstask.sh and *_calibrate.sh.

The generated scripts are not committed; this generator is. Edit the constants
below and rerun to produce a new sweep:

    uv run slurm/generate_slurm.py                 # the sweep
    uv run slurm/generate_slurm.py --calibrate     # the calibration runs

Set SLURM_ACCOUNT to name an account in the generated scripts; without it they
submit under the user's default.
"""
import argparse
import os
from pathlib import Path

SLURM_DIR = Path(__file__).parent

# --- experiment matrix ---------------------------------------------------
# Every (task, memory, seed) combination in TASKS x BASELINE_MEMORIES x SEEDS
# (plus the intrinsic ablations) is run once per experiment script, and every
# (task, intrinsic arm, seed) combination once per crosstask script.

TASKS = ["babyai", "fever", "hotpotqa", "jericho", "pddl", "sciworld"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

# --- calibration ---------------------------------------------------------
# One dataset, every arm, one seed, twenty tasks at the full trial budget. This
# is the run that says whether a 24-hour job fits: the result rows carry the
# token spend, so tokens / elapsed = throughput, and
# episodes x tokens-per-task / throughput = wall clock. Twenty also takes
# g-memory past its twentieth task, where merge_insights runs.

CALIBRATION_SEED = SEEDS[0]
CALIBRATION_TASKS = 20
CALIBRATION_TIME_LIMIT = "02:00:00"

# Tasks whose full budget will not calibrate inside that window, and the smaller
# shakedown that will. Jericho runs 100 trials rather than 30 and its prompt
# tokens grow with the square of the budget - about 1.44M per task by the curve
# in data/data.md, so twenty tasks over ten arms is ~288M tokens, an 18-hour job
# at the throughput the other calibrations measured. Cut to the 2-hour window it
# would report a tenth of its arms and nothing about the rest, which reads as an
# arm that failed rather than one that never ran. Five tasks at 20 trials is
# ~8M tokens and answers what a calibration of Jericho can: whether it runs.
# Sizing its real job needs a job of its own.
CALIBRATION_OVERRIDES = {"jericho": {"max_tasks": 5, "max_trials": 20}}

# Non-intrinsic baselines, run in the experiment script only.
BASELINE_MEMORIES = [
    "empty", "chatdev", "voyager", "memorybank", "generative", "metagpt", "g-memory",
]
# Intrinsic ablations that don't vary by task; the per-task intrinsic memory
# (intrinsicmemory-<task>) is added alongside these in both scripts.
INTRINSIC_ABLATIONS = ["intrinsicmemory-notemplate", "intrinsicmemory-llm-structured-template"]

# vLLM's request queue depth, per task; tasks not listed use DEFAULT_MAX_NUM_SEQS.
MAX_NUM_SEQS_OVERRIDES = {"pddl": 256}
DEFAULT_MAX_NUM_SEQS = 512

# The budget a call starts at, per task; tasks not listed use DEFAULT_MAX_TOKENS.
# A starved reasoning model is retried with a doubled budget, so too small a
# start is paid for in whole wasted calls rather than in a truncated answer.
MAX_TOKENS_OVERRIDES = {"babyai": 4096, "pddl": 4096}
DEFAULT_MAX_TOKENS = 2048

# --- cluster / model configuration ---------------------------------------

REPO_DIR = "~/GMemory"
VLLM_VENV_DIR = "~/vllm_test"
YAML_CONFIG = "/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml"
HF_HOME = "/projects/public/brics/hf"
MODEL_SNAPSHOT = "b5c939de8f754692c1647ca79fbf85e8c1e70f8a"
MODEL_PATH = f"{HF_HOME}/hub/models--openai--gpt-oss-120b/snapshots/{MODEL_SNAPSHOT}/"
MODEL_NAME = "openai/gpt-oss-120b"
TIKTOKEN_ENCODINGS_BASE = "/projects/public/brics/distributed_vllm/etc/encodings"
DEFAULT_DB_DIR = "$HOME/GMemory/.db/sweep"

NODES = 1
GPUS = 4
TENSOR_PARALLEL_SIZE = 4
CPUS_PER_TASK = 16
TIME_LIMIT = "24:00:00"
PORT = 8000
VLLM_STARTUP_SLEEP = 100

# Named at generation time rather than committed: it is one user's allocation.
ACCOUNT = os.environ.get("SLURM_ACCOUNT")

# These three override the shared GPT-OSS_Hopper.yaml, which sets them to 8192,
# 10240 and off.
#
# MAX_NUM_BATCHED_TOKENS is a per-step total across the whole batch, not a
# per-request cap, so being the larger of the two is what lets a step prefill
# several requests at once instead of one at a time. MAX_MODEL_LEN is per
# request, and has to clear the largest prompt plus --max_tokens_ceiling.
MAX_NUM_BATCHED_TOKENS = 32768
MAX_MODEL_LEN = 16384
ENABLE_PREFIX_CACHING = True

# One worker process per experiment, each loading an embedding model on the CPU.
# Unset, every one of them sizes its thread pool to the whole node.
OMP_NUM_THREADS = 2

# --------------------------------------------------------------------------


def intrinsic_memory_for(task: str) -> str:
    return f"intrinsicmemory-{task}"


def vllm_serve_block(max_num_seqs: int) -> str:
    return f"""cd {VLLM_VENV_DIR}

source .venv/bin/activate

YAML_CONFIG="{YAML_CONFIG}"
HF_HOME={HF_HOME}
MODEL_PATH=$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/{MODEL_SNAPSHOT}/
MODEL_NAME="{MODEL_NAME}"

export TIKTOKEN_ENCODINGS_BASE="{TIKTOKEN_ENCODINGS_BASE}"

srun \\
    --nodes=$SLURM_NNODES \\
    --gpus=$SLURM_GPUS \\
    --cpus-per-task {CPUS_PER_TASK} \\
    --ntasks-per-node 1 \\
    vllm serve $MODEL_PATH \\
    --served-model-name $MODEL_NAME \\
    --config $YAML_CONFIG \\
    --host 0.0.0.0 \\
    --port {PORT} \\
    --max-num-seqs {max_num_seqs} \\
    --max-num-batched-tokens {MAX_NUM_BATCHED_TOKENS} \\
    --max-model-len {MAX_MODEL_LEN} \\
    {"--enable-prefix-caching" if ENABLE_PREFIX_CACHING else "--no-enable-prefix-caching"} \\
    --tensor_parallel_size={TENSOR_PARALLEL_SIZE} &

VLLM_PID=$!

# wait for vllm to start up
until curl -s http://localhost:{PORT}/health > /dev/null 2>&1; do
  echo "Waiting for vLLM to be ready..."
  sleep 5
done

echo "vLLM started!"
curl -s http://localhost:{PORT}/v1/models

deactivate"""


CALIBRATION_SUMMARY = """
echo "==== calibration ===="
column -s, -t < ${DB_DIR}/overall_results.csv

python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tokens = sum(int(r["completion_tokens"]) + int(r["prompt_tokens"]) for r in rows)
scored = sum(int(r["tasks_scored"]) for r in rows)
print(f"{len(rows)} arms, {scored} tasks scored, {tokens:,} tokens")
print(f"{tokens/max(scored, 1):,.0f} tokens per task")
' ${DB_DIR}/overall_results.csv

cat ${DB_DIR}/*/*/*/*/failed_tasks.csv 2>/dev/null
"""


def account_directive() -> str:
    return f"#SBATCH --account={ACCOUNT}\n" if ACCOUNT else ""


def render(task: str, variant: str) -> str:
    script_name = f"{task}_{variant}.sh"
    max_num_seqs = MAX_NUM_SEQS_OVERRIDES.get(task, DEFAULT_MAX_NUM_SEQS)
    time_limit = CALIBRATION_TIME_LIMIT if variant == "calibrate" else TIME_LIMIT
    max_tokens = MAX_TOKENS_OVERRIDES.get(task, DEFAULT_MAX_TOKENS)
    cross_task_comment = ""
    cross_task_flag = ""
    summary = ""

    if variant == "crosstask":
        job_name = f"vllm-{task}-cross"
        output_pattern = f"out/{task}-cross-%x.%j.%t.out"
        memories = " ".join(
            [INTRINSIC_ABLATIONS[0], intrinsic_memory_for(task), INTRINSIC_ABLATIONS[1]]
        )
        seeds = " ".join(str(s) for s in SEEDS)
        scope = ""
        db_dir = DEFAULT_DB_DIR
        cross_task_comment = """
# The cross-task arm: an intrinsic memory is kept across the tasks of the dataset
# instead of starting each task from an empty one. Only the intrinsicmemory-* modules
# read the flag, and the same --db_dir as the baseline is deliberate - the two arms are
# told apart by the intrinsic_cross_task column, not by the file they are in.
"""
        cross_task_flag = "\n\t--intrinsic_cross_task \\"
    elif variant == "calibrate":
        job_name = f"vllm-{task}-calibrate"
        output_pattern = f"out/{task}-calibrate-%x.%j.%t.out"
        memories = " ".join(
            BASELINE_MEMORIES
            + [INTRINSIC_ABLATIONS[0], intrinsic_memory_for(task), INTRINSIC_ABLATIONS[1]]
        )
        seeds = str(CALIBRATION_SEED)
        overrides = CALIBRATION_OVERRIDES.get(task, {})
        scope = f"\n\t--max_tasks {overrides.get('max_tasks', CALIBRATION_TASKS)} \\"
        if "max_trials" in overrides:
            scope += f"\n\t--max_trials {overrides['max_trials']} \\"
        db_dir = f"{DEFAULT_DB_DIR}/calibration"
        summary = CALIBRATION_SUMMARY
    else:
        job_name = f"vllm-{task}"
        output_pattern = f"out/{task}-%x.%j.%t.out"
        memories = " ".join(
            BASELINE_MEMORIES
            + [INTRINSIC_ABLATIONS[0], intrinsic_memory_for(task), INTRINSIC_ABLATIONS[1]]
        )
        seeds = " ".join(str(s) for s in SEEDS)
        scope = ""
        db_dir = DEFAULT_DB_DIR

    return f"""#!/bin/bash
{account_directive()}#SBATCH --job-name={job_name}
#SBATCH --nodes={NODES}
#SBATCH --gpus={GPUS}
#SBATCH --time={time_limit}
#SBATCH --exclusive
#SBATCH --output={output_pattern}

echo SERVING ON $HOSTNAME

module reset
module load brics/nccl
module list

# Every job of one experiment set must point at the same directory: they append to
# one overall_results.csv under a lock on the file. Override at submit time with
#   DB_DIR=/projects/<project>/results/sweep-2026-09 sbatch slurm/{script_name}
DB_DIR=${{DB_DIR:-{db_dir}}}

{vllm_serve_block(max_num_seqs)}

# experiment setup
export MODEL_NAME="{MODEL_NAME}"
export OPENAI_API_BASE=http://localhost:{PORT}/v1
export OPENAI_API_KEY="none"
export OMP_NUM_THREADS={OMP_NUM_THREADS}

cd {REPO_DIR}
source .venv/bin/activate

sleep {VLLM_STARTUP_SLEEP}

echo "results -> ${{DB_DIR}}"
{cross_task_comment}
# --no-sync: several of these jobs share one .venv, and `uv run` would otherwise
# resolve and relink it under whichever of them is already importing torch.
uv run --no-sync tasks/run.py \\
\t--task {task} \\
\t--mas_type autogen \\
\t--mas_memory {memories} \\
\t--seed {seeds} \\{cross_task_flag}{scope}
\t--db_dir ${{DB_DIR}} \\
\t--model ${{MODEL_NAME}} \\
\t--max_tokens {max_tokens}
{summary}
# cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--calibrate",
        action="store_true",
        help="generate the calibration scripts instead of the sweep",
    )
    parser.add_argument(
        "--task",
        nargs="+",
        choices=TASKS,
        default=TASKS,
        help="the datasets to generate for (default: all of them)",
    )
    args = parser.parse_args()

    variants = ("calibrate",) if args.calibrate else ("experiment", "crosstask")
    for task in args.task:
        for variant in variants:
            path = SLURM_DIR / f"{task}_{variant}.sh"
            path.write_text(render(task, variant))
            path.chmod(0o755)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
