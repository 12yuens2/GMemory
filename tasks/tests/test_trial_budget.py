"""Each task's size comes from its own entry in tasks/configs.yaml.

Two numbers, the same shape: how many trials one episode gets (`max_steps`) and
how many tasks of the dataset are covered (`max_tasks`). `--max_trials` and
`--max_tasks` override each for every task in the sweep, so the config is the
only place a per-task value can be expressed.
"""

from pathlib import Path

import pytest
import yaml

from tasks.envs import ENVS
from tasks.tests.fakes import FakeEnv

with open("tasks/configs.yaml") as reader:
    TASK_SETTINGS: dict = yaml.safe_load(reader)

# The env registry is the list of tasks that exist; a task registered without a
# budget in tasks/configs.yaml is what test_every_task_declares_a_budget catches.
TASKS = sorted(ENVS)


@pytest.fixture
def run_module(monkeypatch):
    """experiment.py with its environment and dataset construction stubbed out.

    `built` collects the trial budget each env was constructed with.
    """
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("experiment")

    built: list[tuple[str, int]] = []

    def fake_get_env(task, config, max_trials):
        built.append((task, max_trials))
        return FakeEnv(max_trials=max_trials)

    monkeypatch.setattr(module, "get_env", fake_get_env)
    monkeypatch.setattr(module, "get_task", lambda task, **kwargs: [{"task": "a task"}])
    module.built = built
    return module


# ── the budget is read per task ───────────────────────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_every_task_declares_a_budget(task):
    assert task in TASK_SETTINGS, (
        f"{task} is a registered environment with no entry in tasks/configs.yaml"
    )
    assert "max_steps" in TASK_SETTINGS[task], (
        f"{task} has no max_steps, so nothing can resolve its episode budget"
    )


@pytest.mark.parametrize("task", TASKS)
def test_the_declared_budget_is_what_resolves(run_module, task):
    assert run_module.trial_budget(task) == TASK_SETTINGS[task]["max_steps"]


@pytest.mark.parametrize("task", TASKS)
def test_the_environment_is_built_with_the_declared_budget(run_module, task, tmp_path):
    run_module.build_task(
        task=task, mas_type="autogen", memory_type="empty",
        seed=42, working_dir=str(tmp_path),
    )

    assert run_module.built == [(task, TASK_SETTINGS[task]["max_steps"])]


def test_two_tasks_in_one_sweep_get_their_own_budgets(run_module, tmp_path, monkeypatch):
    monkeypatch.setattr(run_module, "CONFIG", {
        "fever": {"max_steps": 7, "env_config_path": "tasks/env_configs/fever_config.yaml"},
        "pddl": {"max_steps": 19, "env_config_path": "tasks/env_configs/pddl_config.yaml"},
    })

    for task in ("fever", "pddl"):
        run_module.build_task(
            task=task, mas_type="autogen", memory_type="empty",
            seed=42, working_dir=str(tmp_path),
        )

    assert run_module.built == [("fever", 7), ("pddl", 19)]


# ── --max_trials is an override, not the source ───────────────────────────────

@pytest.mark.parametrize("task", TASKS)
def test_an_explicit_override_wins_for_every_task(run_module, task):
    assert run_module.trial_budget(task, 5) == 5


def test_an_override_reaches_the_environment(run_module, tmp_path):
    run_module.build_task(
        task="fever", mas_type="autogen", memory_type="empty",
        seed=42, working_dir=str(tmp_path), max_trials=4,
    )

    assert run_module.built == [("fever", 4)]


def test_a_task_with_no_configured_budget_says_so(run_module, monkeypatch):
    monkeypatch.setattr(run_module, "CONFIG", {"fever": {}})

    with pytest.raises(KeyError, match="max_steps"):
        run_module.trial_budget("fever")


# ── how many tasks, the same shape as the trial budget ────────────────────────

def test_the_configured_number_of_tasks_is_what_resolves(run_module, monkeypatch):
    monkeypatch.setattr(run_module, "CONFIG", {"fever": {"max_steps": 30, "max_tasks": 5}})

    assert run_module.dataset_size("fever") == 5


def test_a_task_with_no_configured_limit_covers_the_whole_dataset(run_module, monkeypatch):
    monkeypatch.setattr(run_module, "CONFIG", {"pddl": {"max_steps": 30}})

    assert run_module.dataset_size("pddl") is None


def test_an_override_wins_over_the_configured_number_of_tasks(run_module):
    assert run_module.dataset_size("fever", 2) == 2


def test_the_override_reaches_the_dataset(run_module, monkeypatch, tmp_path):
    """The point of it: a smoke run has to be short without editing the config."""
    asked_for: list = []
    monkeypatch.setattr(
        run_module, "get_task",
        lambda task, **kwargs: asked_for.append(kwargs.get("max_tasks")) or [{"task": "a task"}],
    )

    run_module.build_task(
        task="fever", mas_type="autogen", memory_type="empty",
        seed=42, working_dir=str(tmp_path), max_tasks=2,
    )

    assert asked_for == [2]
