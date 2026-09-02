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

## Jericho

The interpreter comes from the `jericho` package; the game files do not, and are
not in this repository. The manifest lists the 56 games of the Jericho suite whose
score Jericho can read, and the roms go beside it in `data/jericho/roms/`:

```
mkdir -p jericho/roms
curl -s "https://api.github.com/repos/BYU-PCCL/z-machine-games/git/trees/master?recursive=1" \
  | jq -r '.tree[].path | select(startswith("jericho-game-suite/")) | select(contains("/") and (split("/") | length == 2))' \
  | while read -r path; do
      curl -sL -o "jericho/roms/$(basename "$path")" \
        "https://raw.githubusercontent.com/BYU-PCCL/z-machine-games/master/${path}"
    done
```

That fetches 57 files, 8.5 MB. One of them, `lgop.z3`, is deliberately not in the
manifest: Jericho reports a maximum score of 0 for it and knows no walkthrough, so
neither the progress rate nor a victory could ever be scored. The other 56 all
load and report a score range.

Four of the games open above zero — `advent` on 36 of 350, `detective` on 10 of
360, `deephome` on 1 of 300 and `ludicorp` on 1 of 150 — which is why the progress
rate is measured from the opening score rather than from nothing.

## BabyAI

Nothing to download. `minigrid` generates each gridworld from a level name and a
seed, so the manifest holds those instead of the data: 20 of the 96 registered
BabyAI levels, spread across the competence ladder, at 10 seeds each for 200 tasks.

It is written seed-major - every level at seed 0, then every level at seed 1 - so
that a `--max_tasks` prefix is a spread across the levels rather than many seeds of
the first one. `tasks/tests/test_babyai_env.py` asserts that ordering.

To regenerate it, or to change the levels or the number of seeds:

```python
import json
import gymnasium as gym
import minigrid  # registers the BabyAI levels

LEVELS = [...]           # level ids, all of which must be in gym.registry
SEEDS = range(10)

with open('babyai/babyai_levels.jsonl', 'w') as out:
    for seed in SEEDS:
        for level in LEVELS:
            out.write(json.dumps({'id': f'{level}-seed{seed}', 'level': level, 'seed': seed}) + '\n')
```

The full list of registered levels is `sorted(k for k in gym.registry if k.startswith('BabyAI-'))`.
