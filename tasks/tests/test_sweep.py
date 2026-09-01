"""Which experiments a sweep runs, and how they are kept apart.

Three properties, all of which fail invisibly rather than loudly:

  - a flag given several values must produce one experiment per combination;
  - concurrent workers must share the lock they append result files through;
  - two seeds of one config must not share a memory persistence directory.

The last one is the expensive kind of wrong: memory carried from one seed into
another does not fail, it just quietly answers a different question.
"""

import importlib
import sys
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace

import pytest

from tasks.envs.base_env import BaseRecorder
from tasks.tests.fakes import FakeEnv
from tasks.tests.test_run_task import StubMAS


@pytest.fixture
def sweep_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("sweep")


@pytest.fixture
def experiment_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("experiment")


def parse(sweep, *argv: str):
    return sweep.build_arg_parser().parse_args(list(argv))


# ── C3 · one experiment per combination ───────────────────────────────────────

def test_every_combination_of_the_swept_flags_is_one_experiment(sweep_module):
    sweep = sweep_module
    args = parse(
        sweep,
        '--mas_type', 'autogen',
        '--task', 'fever', 'pddl',
        '--mas_memory', 'empty', 'voyager', 'g-memory',
        '--seed', '1', '2', '3',
    )

    configs = sweep.build_experiment_configs(args)

    assert len(configs) == 2 * 3 * 3, "2 tasks x 3 memory modules x 3 seeds"
    distinct = {(config['task'], config['mas_memory'], config['seed']) for config in configs}
    assert len(distinct) == len(configs), "the same experiment appears more than once"


def test_one_value_per_flag_is_one_experiment(sweep_module):
    sweep = sweep_module
    args = parse(sweep, '--mas_type', 'autogen', '--task', 'fever', '--mas_memory', 'empty')

    assert len(sweep.build_experiment_configs(args)) == 1


# ── C1 · running them, in this process or in a pool ───────────────────────────

class FakeExecutor:
    """ProcessPoolExecutor's shape, running each submission inline."""

    def __init__(self, log: list, max_workers=None, mp_context=None):
        self.log = log
        self.max_workers = max_workers

    def __enter__(self):
        return self

    def __exit__(self, *exc_info):
        return False

    def submit(self, function, *args):
        self.log.append(args)
        future = Future()
        future.set_result(function(*args))
        return future


class FakeManager:
    """multiprocessing.Manager()'s shape, handing out one lock."""

    def __init__(self, lock):
        self.lock = lock

    def __enter__(self):
        return SimpleNamespace(Lock=lambda: self.lock)

    def __exit__(self, *exc_info):
        return False


def fake_multiprocessing(lock):
    return SimpleNamespace(get_context=lambda method: method, Manager=lambda: FakeManager(lock))


def test_a_single_experiment_does_not_spawn_a_pool(sweep_module, monkeypatch):
    sweep = sweep_module
    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {'ran': config['seed']})
    monkeypatch.setattr(sweep, 'ProcessPoolExecutor', _refuse_to_be_used)

    assert sweep.run_experiments([{'seed': 42}], num_workers=8) == [{'ran': 42}]


def test_one_worker_runs_every_experiment_in_this_process(sweep_module, monkeypatch):
    sweep = sweep_module
    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {'ran': config['seed']})
    monkeypatch.setattr(sweep, 'ProcessPoolExecutor', _refuse_to_be_used)

    outcomes = sweep.run_experiments([{'seed': 1}, {'seed': 2}], num_workers=1)

    assert outcomes == [{'ran': 1}, {'ran': 2}]


def test_every_parallel_worker_is_given_the_shared_output_lock(sweep_module, monkeypatch):
    """The workers append to the same result files, so the lock has to reach all of them."""
    sweep = sweep_module
    lock = object()
    submissions: list[tuple] = []

    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {'ran': config['seed']})
    monkeypatch.setattr(sweep, 'multiprocessing', fake_multiprocessing(lock))
    monkeypatch.setattr(
        sweep, 'ProcessPoolExecutor', lambda **kwargs: FakeExecutor(submissions, **kwargs)
    )

    experiments = [{'seed': seed} for seed in (1, 2, 3)]
    outcomes = sweep.run_experiments(experiments, num_workers=2)

    assert [config for config, _ in submissions] == experiments, "every experiment is submitted once"
    assert {passed_lock for _, passed_lock in submissions} == {lock}
    assert sorted(outcome['ran'] for outcome in outcomes) == [1, 2, 3]


def test_no_more_workers_are_started_than_there_are_experiments(sweep_module, monkeypatch):
    sweep = sweep_module
    started: list = []

    monkeypatch.setattr(sweep, 'run_experiment', lambda config, *args: {})
    monkeypatch.setattr(sweep, 'multiprocessing', fake_multiprocessing(object()))
    monkeypatch.setattr(
        sweep, 'ProcessPoolExecutor', lambda **kwargs: FakeExecutor(started, **kwargs)
    )

    sweep.run_experiments([{'seed': 1}, {'seed': 2}], num_workers=32)

    assert len(started) == 2


def _refuse_to_be_used(*args, **kwargs):
    raise AssertionError("a worker pool was spawned for work that runs in this process")


# ── C2 · two seeds of one config are kept apart ───────────────────────────────

def build_config(tmp_path, seed: int) -> dict:
    return {
        'task': 'fever', 'mas_type': 'autogen', 'mas_memory': 'empty', 'reasoning': 'io',
        'model': 'fake-model', 'max_trials': 3, 'seed': seed, 'successful_topk': 1,
        'failed_topk': 0, 'insights_topk': 3, 'threshold': 0.0, 'use_projector': False,
        'use_validator': False, 'hop': 1, 'num_workers': 1, 'db_dir': str(tmp_path),
    }


def test_two_seeds_of_one_config_do_not_share_a_memory_directory(
    experiment_module, monkeypatch, tmp_path
):
    experiment = experiment_module
    memory_dirs: list[str] = []

    def fake_build_task(task, mas_type, memory_type, seed, working_dir, model=None, max_trials=None):
        env = FakeEnv(max_trials=max_trials or 3)
        return experiment.TaskManager(
            task_name=task,
            mas_type=mas_type,
            memory_type=memory_type,
            tasks=[],
            env=env,
            recorder=BaseRecorder(working_dir=working_dir, namespace=f'seed_{seed}'),
            mas=StubMAS(env),
            seed=seed,
            model=model,
            token_tracker=experiment.TokenTracker(),
        )

    monkeypatch.setattr(experiment, 'build_task', fake_build_task)
    monkeypatch.setattr(experiment, 'build_mas', lambda *args, **kwargs: None)
    monkeypatch.setattr(
        experiment,
        'run_task',
        lambda manager, working_dir, output_lock=None: memory_dirs.append(
            manager.mem_config['working_dir']
        ),
    )

    for seed in (42, 43):
        outcome = experiment.run_experiment(build_config(tmp_path, seed))
        assert outcome['status'] == 'success', outcome.get('error')

    assert len(set(memory_dirs)) == 2, (
        f"both seeds persisted their memory to the same place: {memory_dirs}"
    )
