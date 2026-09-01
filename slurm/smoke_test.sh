#!/bin/bash
#SBATCH --job-name=vllm-smoke
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:30:00
#SBATCH --exclusive
#SBATCH --output=out/smoke-%x.%j.%t.out

# The whole path, small: serve, run two tasks of one dataset through two memory
# modules, and check what came out. Run this before submitting a 24-hour job, and
# after any change to the cluster, the model or the environment.

set -euo pipefail

echo SERVING ON $HOSTNAME

module reset
module load brics/nccl
module list

MODEL_NAME="openai/gpt-oss-120b"
YAML_CONFIG="/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml"
HF_HOME=/projects/public/brics/hf
MODEL_PATH=$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/
DB_DIR=./.db/smoke-${SLURM_JOB_ID:-local}

export TIKTOKEN_ENCODINGS_BASE="/projects/public/brics/distributed_vllm/etc/encodings"

cd ~/vllm_test
source .venv/bin/activate

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

until curl -s http://localhost:8000/health > /dev/null 2>&1; do
  echo "Waiting for vLLM to be ready..."
  sleep 5
done

echo "vLLM started!"
deactivate

export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY="none"

cd ~/GMemory
source .venv/bin/activate

# The result files are appended to under an flock, which a filesystem has to be
# mounted for. Lustre supports it with the flock mount option; without it,
# separately submitted jobs can interleave their writes.
mkdir -p ${DB_DIR}
if flock -n ${DB_DIR}/.flock-probe true 2>/dev/null; then
  echo "flock: supported on $(df -T ${DB_DIR} 2>/dev/null | tail -1 || df ${DB_DIR} | tail -1)"
else
  echo "flock: REFUSED - jobs writing to one results file may interleave"
fi

uv run tasks/run.py \
	--task fever \
	--mas_type autogen \
	--mas_memory empty intrinsicmemory-notemplate \
	--seed 11 \
	--max_tasks 2 \
	--max_trials 3 \
	--db_dir ${DB_DIR} \
	--model ${MODEL_NAME}

echo "==== what the run wrote ===="
find ${DB_DIR} -name '*.csv' | sort
echo "==== overall_results.csv ===="
cat ${DB_DIR}/overall_results.csv

# 2 memory modules, so 2 rows plus a header. Fewer means an experiment failed.
rows=$(($(wc -l < ${DB_DIR}/overall_results.csv) - 1))
if [ "$rows" -ne 2 ]; then
  echo "SMOKE TEST FAILED: expected 2 result rows, got ${rows}"
  cat ${DB_DIR}/failed_experiments.csv 2>/dev/null
  kill $VLLM_PID 2>/dev/null
  exit 1
fi

echo "SMOKE TEST PASSED"

kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
