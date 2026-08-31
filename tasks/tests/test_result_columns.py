"""`use_validator` is recorded alongside the result it produced.

`working_dir` is keyed on `mas_type`, and `use_validator` is not part of it, so
without a column a validator run and a plain run of the same task, memory and
model are indistinguishable once written.

The column is appended last in every file that has headers, so a reader indexing
the existing columns positionally is unaffected.
"""

import csv
from pathlib import Path

import pytest


@pytest.fixture
def run_module():
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("run")


def experiment(**overrides) -> dict:
    config = {
        'task': 'fever', 'mas_type': 'autogen', 'mas_memory': 'empty',
        'model': 'gpt-oss-120b', 'seed': 42, 'max_trials': 30, 'num_workers': 1,
        'use_validator': False,
    }
    config.update(overrides)
    return config


def rows(path: Path) -> list[dict]:
    with path.open() as reader:
        return list(csv.DictReader(reader))


# ── overall_results.csv, the aggregate ────────────────────────────────────────

@pytest.mark.parametrize('use_validator', [False, True])
def test_the_aggregate_records_whether_the_validator_ran(run_module, tmp_path, use_validator):
    result = "gpt-oss-120b,fever,empty,0.5,0.5,3.0,10,20,0,0,42\n"

    run_module._write_overall_result(
        experiment(use_validator=use_validator), result, str(tmp_path)
    )

    written = rows(tmp_path / 'overall_results.csv')
    assert written[0]['use_validator'] == str(use_validator), (
        f"use_validator column reads {written[0]['use_validator']!r}"
    )


def test_the_new_column_is_last(run_module, tmp_path):
    """A reader indexing the previous columns positionally must be unaffected."""
    result = "gpt-oss-120b,fever,empty,0.5,0.5,3.0,10,20,0,0,42\n"

    run_module._write_overall_result(experiment(), result, str(tmp_path))

    header = (tmp_path / 'overall_results.csv').read_text().splitlines()[0].split(',')
    assert header[-1] == 'use_validator', f"header ends {header[-3:]}"
    assert header[:7] == ['task', 'mas_type', 'mas_memory', 'model', 'seed',
                          'max_trials', 'num_workers'], "the leading columns moved"


def test_the_row_has_one_value_per_header(run_module, tmp_path):
    result = "gpt-oss-120b,fever,empty,0.5,0.5,3.0,10,20,0,0,42\n"

    run_module._write_overall_result(experiment(), result, str(tmp_path))

    lines = (tmp_path / 'overall_results.csv').read_text().strip().splitlines()
    assert len(lines[0].split(',')) == len(lines[1].split(',')), (
        f"{len(lines[0].split(','))} headers, {len(lines[1].split(','))} values"
    )


# ── failed_experiments.csv ────────────────────────────────────────────────────

@pytest.mark.parametrize('use_validator', [False, True])
def test_a_failed_experiment_records_whether_the_validator_ran(
    run_module, tmp_path, use_validator
):
    run_module._write_failed_experiment(
        experiment(use_validator=use_validator), ValueError('no endpoint'), str(tmp_path)
    )

    written = rows(tmp_path / 'failed_experiments.csv')
    assert written[0]['use_validator'] == str(use_validator)


# ── failed_tasks.csv ──────────────────────────────────────────────────────────

@pytest.mark.parametrize('use_validator', [False, True])
def test_a_failed_task_records_whether_the_validator_ran(
    run_module, tmp_path, use_validator
):
    from tasks.envs.base_env import BaseRecorder

    manager = run_module.TaskManager(
        task_name='fever', mas_type='autogen', memory_type='empty', tasks=[],
        env=None, recorder=BaseRecorder(working_dir=str(tmp_path), namespace='n'),
        mas=None, token_tracker=run_module.TokenTracker(),
    )
    manager.mas_config = {'use_validator': use_validator}

    run_module._write_failed_tasks(
        manager, 42, [{'task_id': 0, 'error': RuntimeError('boom')}], str(tmp_path)
    )

    written = rows(tmp_path / 'failed_tasks.csv')
    assert written[0]['use_validator'] == str(use_validator)
