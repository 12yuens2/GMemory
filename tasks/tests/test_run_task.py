"""run_task's loop over a dataset (test plan group C, partial).

The sweep runner had no tests. Every fixture here drives the real run_task with a
fake environment, workflow and LLM, so the loop, the failure handling and the
per-task CSV row are exercised rather than described.

The failure behaviour these tests pin is deliberately split in two:

  - an agent that cannot produce an action is handled inside schedule, where the
    trial count is known, and the task IS scored - as unsolved;
  - anything around the episode (set_env, prompt assembly, the recorder) is
    recorded and NOT scored, because an environment fault is not an episode
    outcome and does not belong in the reward column.

Either way the next task in the dataset runs. Before this, both killed the whole
experiment.
"""

import csv
from pathlib import Path

import pytest

from mas.mas import EpisodeResult

from tasks.envs.base_env import BaseRecorder
from tasks.tests.fakes import FakeEnv


class StubMAS:
    """Stands in for a built workflow: run_task only reaches .env, .agents_team
    and .schedule."""

    def __init__(self, env, outcomes=None, fail_on=()):
        self.env = env
        self.agents_team = {}
        self.outcomes = outcomes
        self.fail_on = set(fail_on)
        self.scheduled: list[dict] = []

    def schedule(self, task_config: dict) -> EpisodeResult:
        index = len(self.scheduled)
        self.scheduled.append(dict(task_config))
        if index in self.fail_on:
            raise RuntimeError(f"workflow exploded on task {index}")
        if self.outcomes:
            return self.outcomes[index % len(self.outcomes)]
        return EpisodeResult(reward=1.0, done=True, trials=1)


class ExplodingEnv(FakeEnv):
    """An environment whose set_env fails for chosen task indices."""

    def __init__(self, fail_on=(), **kwargs):
        super().__init__(**kwargs)
        self.fail_on = set(fail_on)
        self.set_env_calls = 0

    def set_env(self, task_config: dict) -> tuple[str, str]:
        index = self.set_env_calls
        self.set_env_calls += 1
        if index in self.fail_on:
            raise RuntimeError(f"simulator unavailable for task {index}")
        return super().set_env(task_config)


@pytest.fixture
def run_task_module(monkeypatch):
    """run.py reads tasks/configs.yaml at import and needs credentials for the
    sweep entry point, neither of which this module's tests should depend on."""
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    module = importlib.import_module("run")
    monkeypatch.setattr(module, "get_task_few_shots", lambda **kwargs: ["a few shot"])
    monkeypatch.setattr(module, "get_dataset_system_prompt", lambda *a, **k: "do the task")
    monkeypatch.setattr(module, "CONFIG", {"fever": {"few_shots_num": 1}})
    return module


def build_manager(run, tmp_path, tasks, mas):
    recorder = BaseRecorder(working_dir=str(tmp_path), namespace="run-task-test")
    return run.TaskManager(
        task_name="fever",
        mas_type="autogen",
        memory_type="empty",
        tasks=tasks,
        env=mas.env,
        recorder=recorder,
        mas=mas,
        token_tracker=run.TokenTracker(),
    )


def read_csv(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with path.open() as reader:
        return list(csv.DictReader(reader))


# ── the happy path ────────────────────────────────────────────────────────────

def test_every_task_in_the_dataset_is_scheduled(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}", "answer": "SUPPORTS"} for i in range(4)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=42, working_dir=str(tmp_path), task_name="fever")

    assert len(mas.scheduled) == 4
    assert manager.recorder.total_rewards == [1.0] * 4


def test_one_result_row_is_written_per_completed_task(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(FakeEnv())
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=42, working_dir=str(tmp_path), task_name="fever")

    rows = (tmp_path / "fever-empty-results.csv").read_text().strip().splitlines()
    assert len(rows) == 3


# ── C4 · one failing task does not kill the experiment ────────────────────────

def test_a_failing_environment_does_not_stop_the_remaining_tasks(run_task_module, tmp_path):
    """A simulator that fails to load one task used to abandon the rest."""
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(5)]
    mas = StubMAS(ExplodingEnv(fail_on=(2,)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=42, working_dir=str(tmp_path), task_name="fever")

    assert len(mas.scheduled) == 4, "the four healthy tasks should all have run"
    assert manager.recorder.total_rewards == [1.0] * 4


def test_a_failing_workflow_does_not_stop_the_remaining_tasks(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(4)]
    mas = StubMAS(FakeEnv(), fail_on=(0, 2))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=42, working_dir=str(tmp_path), task_name="fever")

    assert len(mas.scheduled) == 4, "every task should have been attempted"
    assert manager.recorder.total_rewards == [1.0, 1.0], "only the two that ran are scored"


def test_a_failed_task_is_recorded_with_its_error(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(ExplodingEnv(fail_on=(1,)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=7, working_dir=str(tmp_path), task_name="fever")

    rows = read_csv(tmp_path / "failed_tasks.csv")
    assert len(rows) == 1
    assert rows[0]["task_id"] == "1"
    assert rows[0]["error_type"] == "RuntimeError"
    assert "simulator unavailable" in rows[0]["error_message"]
    assert rows[0]["seed"] == "7"


def test_a_failed_task_is_excluded_from_the_averages_not_scored_as_zero(
    run_task_module, tmp_path
):
    """An environment fault is not an episode outcome. Scoring it 0 would put
    infrastructure noise in the reward column."""
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(ExplodingEnv(fail_on=(0,)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=42, working_dir=str(tmp_path), task_name="fever")

    rewards, dones, _ = manager.recorder.average_results()
    assert (rewards, dones) == (1.0, 1.0), "the mean is over the two tasks that ran"
    assert len(manager.recorder.total_rewards) == 2


def test_the_exclusion_is_stated_loudly(run_task_module, tmp_path, capsys):
    """results.csv does not say how many tasks are behind its mean, so the log
    and stderr have to."""
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(4)]
    mas = StubMAS(ExplodingEnv(fail_on=(0, 3)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=42, working_dir=str(tmp_path), task_name="fever")

    assert "2/4 tasks failed" in capsys.readouterr().err


def test_a_dataset_where_every_task_fails_still_finishes(run_task_module, tmp_path):
    run = run_task_module
    tasks = [{"task": f"claim {i}"} for i in range(3)]
    mas = StubMAS(ExplodingEnv(fail_on=(0, 1, 2)))
    manager = build_manager(run, tmp_path, tasks, mas)

    run.run_task(manager, seed=42, working_dir=str(tmp_path), task_name="fever")

    assert manager.recorder.average_results() == (0, 0, 0)
    assert len(read_csv(tmp_path / "failed_tasks.csv")) == 3
