#!/bin/bash
#SBATCH --job-name=vllm-serve
#SBATCH --nodes=1
#SBATCH --gpus=4
#SBATCH --time=8:00:00
#SBATCH --exclusive
#SBATCH --output=out/%x.%j.out

if [ $# -ne 1 ]; then
    echo "Usage: $0 <ray_jobid>"
    echo "Example: $0 160852"
    exit 1
fi

RAY_JOBID=$1
HEAD_NODE=$(scontrol show hostnames $(squeue -j ${RAY_JOBID} -h -o %R) | head -n 1)
HEAD_NODE_IP=$(dig +short ${HEAD_NODE})

module reset
module load brics/nccl
module list

export MODEL_NAME="openai/gpt-oss-120b"
export OPENAI_API_BASE=${HEAD_NODE_IP}:8000
export OPENAI_API_KEY="none"

cd ~/GMemory
source .venv/bin/activate

srun uv run tasks/run.py \
	--task sciworld \
	--mas_type autogen \
	--mas_memory g-memory \
	--model ${MODEL_NAME}
