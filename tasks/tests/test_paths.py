"""Where a run reads its configuration from, and where it writes its results.

Both are cluster failures rather than local ones: a job starts in whatever
directory the scheduler gives it, and two models whose names nobody listed must
not share a results directory.
"""

import importlib
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def experiment_module():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("experiment")


def test_the_task_config_and_datasets_resolve_from_the_repository_root(experiment_module):
    """A job's working directory is the scheduler's choice, not the repo."""
    from tasks.envs import TASKS_PATH

    assert Path(experiment_module.CONFIG_PATH).is_absolute()
    assert Path(experiment_module.CONFIG_PATH).exists()

    for task, path in TASKS_PATH.items():
        assert Path(path).is_absolute(), f"{task}'s dataset path is relative: {path}"
        assert Path(path).is_relative_to(REPO_ROOT), f"{task}'s dataset is outside the repo: {path}"


# ── the results directory names the model that produced them ──────────────────

def test_every_model_gets_its_own_results_directory():
    from tasks.utils import model_dir_name

    assert model_dir_name('Qwen/Qwen3.6-35B-A3B') != model_dir_name('openai/gpt-oss-120b')


def test_a_model_name_is_one_directory_and_not_a_path():
    """A served name has a slash in it, which would otherwise nest a directory."""
    from tasks.utils import model_dir_name

    for model_name in ('openai/gpt-oss-120b', '../../etc', 'a b:c'):
        name = model_dir_name(model_name)
        assert Path(name).name == name, f"{model_name!r} became a path: {name!r}"
        assert name not in ('', '.', '..')
