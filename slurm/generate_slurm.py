#!/usr/bin/env python3
"""Generate the per-task slurm/*_experiment.sh and slurm/*_crosstask.sh sweep scripts.

The generated scripts are not committed; this generator is. Edit the constants
below and rerun to produce a new sweep:

    uv run slurm/generate_slurm.py
"""
from pathlib import Path

SLURM_DIR = Path(__file__).parent

# --- experiment matrix ---------------------------------------------------
# Every (task, memory, seed) combination in TASKS x BASELINE_MEMORIES x SEEDS
# (plus the intrinsic ablations) is run once per experiment script, and every
# (task, intrinsic arm, seed) combination once per crosstask script.

TASKS = ["babyai", "fever", "hotpotqa", "jericho", "pddl", "sciworld"]
SEEDS = [11, 22, 33, 44, 55, 66, 77, 88, 99, 111]

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

# Wall clock, per task; tasks not listed use DEFAULT_TIME_LIMIT. A job that runs
# out loses the arms still in flight, and the intrinsic arms are always the ones
# still in flight: they make a summariser call per episode on top of the solver's.
TIME_LIMIT_OVERRIDES = {"babyai": "24:00:00"}
DEFAULT_TIME_LIMIT = "12:00:00"

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
PORT = 8000
VLLM_STARTUP_SLEEP = 100

# These three override the shared GPT-OSS_Hopper.yaml, which sets them to 8192,
# 10240 and off. Prompt tokens are 74% of the sweep's bill and every trial resends
# the same system prompt and few-shots, so prefix caching is the largest single
# saving available; at 8192 batched tokens one 4k prefill fills half a scheduler
# step, which is what held prefill to under 5k tokens/s. MAX_MODEL_LEN has to
# clear the largest prompt plus the budget a retry can climb to.
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
# The flags above are given after --config, which is what makes them win. The
# card echoes max_model_len, so a run whose overrides were dropped says so here
# rather than in a 400 an hour later.
curl -s http://localhost:{PORT}/v1/models

deactivate"""


def render(task: str, crosstask: bool) -> str:
    variant = "crosstask" if crosstask else "experiment"
    script_name = f"{task}_{variant}.sh"
    job_name = f"vllm-{task}-cross" if crosstask else f"vllm-{task}"
    output_pattern = f"out/{task}-cross-%x.%j.%t.out" if crosstask else f"out/{task}-%x.%j.%t.out"
    max_num_seqs = MAX_NUM_SEQS_OVERRIDES.get(task, DEFAULT_MAX_NUM_SEQS)
    max_tokens = MAX_TOKENS_OVERRIDES.get(task, DEFAULT_MAX_TOKENS)
    time_limit = TIME_LIMIT_OVERRIDES.get(task, DEFAULT_TIME_LIMIT)
    seeds = " ".join(str(s) for s in SEEDS)

    if crosstask:
        memories = " ".join(
            [INTRINSIC_ABLATIONS[0], intrinsic_memory_for(task), INTRINSIC_ABLATIONS[1]]
        )
        cross_task_comment = """
# The cross-task arm: an intrinsic memory is kept across the tasks of the dataset
# instead of starting each task from an empty one. Only the intrinsicmemory-* modules
# read the flag, and the same --db_dir as the baseline is deliberate - the two arms are
# told apart by the intrinsic_cross_task column, not by the file they are in.
"""
        cross_task_flag = "\n\t--intrinsic_cross_task \\"
    else:
        memories = " ".join(
            BASELINE_MEMORIES
            + [INTRINSIC_ABLATIONS[0], intrinsic_memory_for(task), INTRINSIC_ABLATIONS[1]]
        )
        cross_task_comment = ""
        cross_task_flag = ""

    return f"""#!/bin/bash
#SBATCH --job-name={job_name}
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
DB_DIR=${{DB_DIR:-{DEFAULT_DB_DIR}}}

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
\t--seed {seeds} \\{cross_task_flag}
\t--db_dir ${{DB_DIR}} \\
\t--model ${{MODEL_NAME}} \\
\t--max_tokens {max_tokens}

# cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
"""


def main() -> None:
    for task in TASKS:
        for crosstask in (False, True):
            script_name = f"{task}_{'crosstask' if crosstask else 'experiment'}.sh"
            path = SLURM_DIR / script_name
            path.write_text(render(task, crosstask))
            path.chmod(0o755)
            print(f"wrote {path}")


if __name__ == "__main__":
    main()
