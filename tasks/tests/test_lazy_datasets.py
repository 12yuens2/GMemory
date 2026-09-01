"""A task's dataset is parsed when it is asked for, not when the registry is imported.

run.py imports the env registry whatever `--task` says, and every spawned worker
imports it again, so a dataset parsed at module scope is work every worker does
four times over for the one task it runs.
"""

import importlib

import pytest

MODULE = "tasks.envs"


@pytest.fixture
def envs():
    """A freshly imported registry, so nothing an earlier test asked for is loaded."""
    return importlib.reload(importlib.import_module(MODULE))


def test_importing_the_registry_parses_no_datasets(envs):
    assert envs._datasets == {}, f"parsed at import: {sorted(envs._datasets)}"
    assert sorted(envs.TASK_LOADERS) == sorted(envs.ENVS), (
        "every registered environment needs a loader for its dataset"
    )


def test_asking_for_one_dataset_parses_only_that_one(envs):
    envs.get_task('fever')

    assert sorted(envs._datasets) == ['fever']


def test_a_dataset_is_parsed_once_however_many_experiments_ask_for_it(envs, monkeypatch):
    first = envs.get_task('fever')

    def explode() -> list:
        raise AssertionError("the dataset was parsed a second time")

    monkeypatch.setitem(envs.TASK_LOADERS, 'fever', explode)

    assert envs.get_task('fever') is first


def test_max_tasks_takes_the_first_n_tasks(envs):
    """FEVER is cut to a fixed prefix, so every run covers the same claims."""
    limited = envs.get_task('fever', max_tasks=5)

    assert limited == envs.get_task('fever')[:5]


def test_an_unregistered_task_is_rejected(envs):
    with pytest.raises(ValueError):
        envs.get_task('not-a-task')
