#!/bin/bash
#SBATCH --job-name=vllm-calibrate
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=out/calibrate-%x.%j.%t.out

echo SERVING ON $HOSTNAME

module reset
module load brics/nccl
module list

# Every job of one experiment set must point at the same directory: they append to
# one overall_results.csv under a lock on the file. Override at submit time with
#   DB_DIR=/projects/<project>/results/sweep-2026-09 sbatch slurm/calibrate.sh
DB_DIR=${DB_DIR:-$HOME/GMemory/.db/sweep}

cd ~/vllm_test

source .venv/bin/activate

YAML_CONFIG="/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml"
HF_HOME=/projects/public/brics/hf
MODEL_PATH=$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/
MODEL_NAME="openai/gpt-oss-120b"

export TIKTOKEN_ENCODINGS_BASE="/projects/public/brics/distributed_vllm/etc/encodings"

srun \
    --nodes=$SLURM_NNODES \
    --gpus=$SLURM_GPUS \
    --cpus-per-task 16 \
    --ntasks-per-node 1 \
    vllm serve $MODEL_PATH \
    --served-model-name $MODEL_NAME \
    --config $YAML_CONFIG \
    --host 0.0.0.0 \
    --port 8000 \
    --max-num-seqs 512 \
    --tensor_parallel_size=4 &

VLLM_PID=$!

# wait for vllm to start up
until curl -s http://localhost:8000/health > /dev/null 2>&1; do
  echo "Waiting for vLLM to be ready..."
  sleep 5
done

echo "vLLM started!"

deactivate

# experiment setup
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY="none"

cd ~/GMemory
source .venv/bin/activate

sleep 100

echo "results -> ${DB_DIR}"

# One dataset, every arm, one seed, twenty tasks at the full trial budget. This is the
# run that says whether a 24-hour job fits: the result rows carry the token spend, so
#   tokens / elapsed = throughput, and episodes x tokens-per-task / throughput = wall clock.
# It also exercises g-memory past its twentieth task, where merge_insights runs.

uv run tasks/run.py \
	--task fever \
	--mas_type autogen \
	--mas_memory empty chatdev voyager memorybank generative metagpt g-memory \
	             intrinsicmemory-notemplate intrinsicmemory-fever \
	             intrinsicmemory-llm-structured-template \
	--seed 11 \
	--max_tasks 20 \
	--db_dir ${DB_DIR}/calibration \
	--model ${MODEL_NAME}

echo "==== calibration ===="
column -s, -t < ${DB_DIR}/calibration/overall_results.csv

python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tokens = sum(int(r["completion_tokens"]) + int(r["prompt_tokens"]) for r in rows)
scored = sum(int(r["tasks_scored"]) for r in rows)
print(f"{len(rows)} arms, {scored} tasks scored, {tokens:,} tokens")
print(f"{tokens/max(scored, 1):,.0f} tokens per task")
' ${DB_DIR}/calibration/overall_results.csv

cat ${DB_DIR}/calibration/*/*/*/*/failed_tasks.csv 2>/dev/null

# cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
