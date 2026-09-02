# Dataset
## ALFWorld

For the tasks suffix json
```
curl -o alfworld_tasks_suffix.json https://raw.githubusercontent.com/LeapLabTHU/ExpeL/e41ec9a24823e7b560c561ab191441b56d9bcefc/data/alfworld/alfworld_tasks_suffix.json
```

For PPDL and game files

```
curl -L -o alfworld.zip https://github.com/alfworld/alfworld/releases/download/0.4.2/json_2.1.3_tw-pddl.zip
```

## PDDL

```
curl -L -o data.tar.gz https://huggingface.co/datasets/hkust-nlp/agentboard/resolve/main/data.tar.gz

tar -zxvf data.tar.gz
```

Get the test.jsonl from data/pddl/test.jsonl

## FEVER
curl -L -o train.jsonl https://fever.ai/download/fever/train.jsonl

## HotpotQA

The original `hotpot_dev_distractor_v1.json` host (`curtis.ml.cmu.edu`) no longer
answers, so the dev split comes from the HuggingFace mirror. Only the question and
the answer are used - the agent searches live Wikipedia rather than a supplied
context - so the manifest keeps the five scalar fields and drops `context` and
`supporting_facts`, which are 45 MB of the 46.

```
for off in $(seq 0 100 7400); do
  curl -s "https://datasets-server.huggingface.co/rows?dataset=hotpotqa%2Fhotpot_qa&config=distractor&split=validation&offset=$off&length=100" \
    | jq -c '.rows[].row | {id, question, answer, level, type}'
done > hotpotqa/hotpotqa_dev.jsonl
```

7405 questions, all `level: hard` (5918 `bridge`, 1487 `comparison`). The API rate
limits partway through, so re-run any offset whose response is not JSON.
