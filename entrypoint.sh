#!/bin/bash

echo "ENTRYPOINT STARTED"
echo "Current dir:"
pwd

echo "Files:"
ls -la

echo "Starting job..."

echo "TASK: $TASK"
echo "MEMORY: $MEMORY"
echo "MODEL: $MODEL_NAME"
echo "SEED: $SEED"
echo "MAS TYPE: $MAS_TYPE"

source env/bin/activate

python tasks/run.py --task "$TASK" --reasoning io --mas_memory "$MEMORY" --mas_type $MAS_TYPE --model "$MODEL_NAME" --seed "$SEED"

echo "Ending job..."
