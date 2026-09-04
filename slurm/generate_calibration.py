#!/usr/bin/env python3
"""Generate the per-task slurm/*_calibrate.sh runs that size the sweep jobs.

One dataset, every arm, one seed, twenty tasks at the full trial budget, in a
2-hour allocation. This is the run that says whether a 24-hour job fits: the
result rows carry the token spend, so tokens / elapsed = throughput, and
episodes x tokens-per-task / throughput = wall clock. Twenty also takes
g-memory past its twentieth task, where merge_insights runs.

    uv run slurm/generate_calibration.py                  # every dataset
    uv run slurm/generate_calibration.py --task fever pddl

The cluster, the model and the arms are generate_slurm.py's; what a calibration
does differently is here.
"""
from generate_slurm import (
    DEFAULT_DB_DIR,
    SEEDS,
    every_arm,
    job_script,
    tasks_from_argv,
    write_script,
)

SEED = SEEDS[0]
MAX_TASKS = 20
TIME_LIMIT = "02:00:00"
DB_DIR = f"{DEFAULT_DB_DIR}/calibration"

# Tasks whose full budget will not calibrate inside that window, and the smaller
# shakedown that will. Jericho runs 100 trials rather than 30 and its prompt
# tokens grow with the square of the budget - about 1.44M per task by the curve
# in data/data.md, so twenty tasks over ten arms is ~288M tokens, an 18-hour job
# at the throughput the other calibrations measured. Cut to the 2-hour window it
# would report a tenth of its arms and nothing about the rest, which reads as an
# arm that failed rather than one that never ran. Five tasks at 20 trials is
# ~8M tokens and answers what a calibration of Jericho can: whether it runs.
# Sizing its real job needs a job of its own.
OVERRIDES = {"jericho": {"max_tasks": 5, "max_trials": 20}}

SUMMARY = """
echo "==== calibration ===="
column -s, -t < ${DB_DIR}/overall_results.csv

python3 -c '
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
tokens = sum(int(r["completion_tokens"]) + int(r["prompt_tokens"]) for r in rows)
scored = sum(int(r["tasks_scored"]) for r in rows)
print(f"{len(rows)} arms, {scored} tasks scored, {tokens:,} tokens")
print(f"{tokens/max(scored, 1):,.0f} tokens per task")
' ${DB_DIR}/overall_results.csv

cat ${DB_DIR}/*/*/*/*/failed_tasks.csv 2>/dev/null
"""


def render(task: str) -> str:
    overrides = OVERRIDES.get(task, {})
    scope = f"\n\t--max_tasks {overrides.get('max_tasks', MAX_TASKS)} \\"
    if "max_trials" in overrides:
        scope += f"\n\t--max_trials {overrides['max_trials']} \\"

    return job_script(
        task=task,
        variant="calibrate",
        tag="-calibrate",
        memories=every_arm(task),
        seeds=[SEED],
        time_limit=TIME_LIMIT,
        db_dir=DB_DIR,
        extra_flags=scope,
        epilogue=SUMMARY,
    )


def main() -> None:
    for task in tasks_from_argv(__doc__):
        write_script(f"{task}_calibrate.sh", render(task))


if __name__ == "__main__":
    main()
