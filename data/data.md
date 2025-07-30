# Dataset
## ALFWorld

Clone ALFWorld repo
```
git clone https://github.com/alfworld/alfworld.git alfworld
cd alfworld
```

Download PPDL and game files

```
export ALFWORLD_DATA=<storage_path>
python scripts/alfworld-download
```

## PDDL

```
curl https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz

tar -zxvf data.tar.gz
```

Get the test.jsonl from data/pddl/test.jsonl

## FEVER
curl -o train.jsonl https://fever.ai/download/fever/train.jsonl
