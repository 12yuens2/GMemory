#!/bin/bash
#SBATCH --job-name=vllm-test
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=00:20:00
#SBATCH --exclusive
#SBATCH --output=out/%x.%j.%t.out

echo SERVING ON $HOSTNAME

module reset
module load brics/nccl
module list

cd ~/vllm_test

source .venv/bin/activate

YAML_CONFIG="/projects/public/brics/distributed_vllm/GPT-OSS_Hopper.yaml"
HF_HOME=/projects/public/brics/hf
#MODEL_PATH=$HF_HOME/hub/models--openai--gpt-oss-120b/snapshots/b5c939de8f754692c1647ca79fbf85e8c1e70f8a/
#MODEL_NAME="openai/gpt-oss-120b"
MODEL_NAME="Qwen/Qwen3.6-35B-A3B"

export TIKTOKEN_ENCODINGS_BASE="/projects/public/brics/distributed_vllm/etc/encodings"

srun \
    --nodes=$SLURM_NNODES \
    --gpus=$SLURM_GPUS \
    --cpus-per-task 16 \
    --ntasks-per-node 1 \
    vllm serve $MODEL_NAME \
    --served-model-name $MODEL_NAME \
    --max-model-len 65536 \
    --enable-prefix-caching \
    --max_num_batched_tokens 8192 \
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

# experiment setup, against the model served above
export MODEL_NAME
export OPENAI_API_BASE=http://localhost:8000/v1
export OPENAI_API_KEY="none"

cd ~/GMemory
source .venv/bin/activate

sleep 100

#memory:
#empty, voyager, memorybank, chatdev, generative, metagpt, g-memory
#intrinsicmemory-pddl, intrinsicmemory-fever, intrinsicmemory-llm-structured-template, intrinsicmemory-notemplate

uv run tasks/run.py \
	--task fever pddl sciworld \
	--mas_type autogen \
	--mas_memory empty voyager g-memory memorybank chatdev generative metagpt intrinsicmemory-notemplate intrinsicmemory-llm-structured-template \
	--seed 11 22 33 44 55 66 77 88 99 111 \
	--model ${MODEL_NAME}

# cleanup
kill $VLLM_PID 2>/dev/null
wait $VLLM_PID 2>/dev/null
