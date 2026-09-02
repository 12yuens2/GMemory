#!/bin/bash
#SBATCH --job-name=vllm-fever-cross
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=24:00:00
#SBATCH --exclusive
#SBATCH --output=out/fever-cross-%x.%j.%t.out

echo SERVING ON $HOSTNAME

module reset
module load brics/nccl
module list

# Every job of one experiment set must point at the same directory: they append to
# one overall_results.csv under a lock on the file. Override at submit time with
#   DB_DIR=/projects/<project>/results/sweep-2026-09 sbatch slurm/fever_crosstask.sh
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

# The cross-task arm: an intrinsic memory is kept across the tasks of the dataset
# instead of starting each task from an empty one. Only the intrinsicmemory-* modules
# read the flag, and the same --db_dir as the baseline is deliberate - the two arms are
# told apart by the intrinsic_cross_task column, not by the file they are in.

uv run tasks/run.py \
	--task fever \
	--mas_type autogen \
	--mas_memory intrinsicmemory-notemplate intrinsicmemory-fever \
	             intrinsicmemory-llm-structured-template \
	--seed 11 22 33 44 55 66 77 88 99 111 \
	--intrinsic_cross_task \
	--db_dir ${DB_DIR} \
	--model ${MODEL_NAME}

# cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
