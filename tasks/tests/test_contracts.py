"""Contract tests across the registries (test plan group D).

Every blocking defect found in the Phase 1 review was one member of a family
disagreeing with its siblings: a recorder with the wrong arity, two workflows
returning two values where the caller unpacks three, a memory module rejecting a
keyword its base class swallowed. Asserting the family contract once, over the
registry, is what catches the next one - a test written against a single
implementation cannot.

These tests deliberately read the registries (`MAS`, `RECORDERS`, `ENVS`) rather
than hard-coded lists, so a new implementation is covered the moment it is
registered.
"""

import numbers
import tempfile

import pytest

from mas.mas import EpisodeResult
from mas.memory import MASMemoryBase

from tasks.envs import RECORDERS

from tasks.mas_workflow import MAS
from tasks.tests.fakes import (
    FakeEmbeddingFunc,
    FakeEnv,
    FakeLLM,
    RecordingObserver,
    fake_reasoning,
)

MEMORY_CONFIG_KEYS = ("successful_topk", "failed_topk", "insights_topk", "threshold")

# One task config per task, carrying whichever keys that task's recorder reads
# out of current_task_config in task_end.
TASK_CONFIGS = {
    "alfworld": {"env_kwargs": {"gamefile": "/data/pick_and_place_simple-1/game.tw-pw"}},
    "fever": {},
    "pddl": {"game_name": "blockworld"},
    "sciworld": {},
}


def build_workflow(mas_type: str, env: FakeEnv, memory_cls=MASMemoryBase, replies=None):
    """Assemble one workflow the way run.py's build_mas does, with fakes."""
    workflow = MAS[mas_type]()
    workflow.add_observer(RecordingObserver())

    llm = FakeLLM(replies=replies)
    memory = memory_cls(
        namespace=f"{mas_type}-contract",
        global_config={"working_dir": tempfile.mkdtemp(), "hop": 1},
        llm_model=llm,
        embedding_func=FakeEmbeddingFunc(),
    )
    workflow.build_system(
        fake_reasoning(replies=replies),
        memory,
        env,
        {key: 1 for key in MEMORY_CONFIG_KEYS} | {"use_projector": False},
    )
    return workflow


# ── D2 · all four workflows return one contract ───────────────────────────────

@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_schedule_returns_the_episode_result_run_py_unpacks(mas_type):
    """schedule() must return what run.py:131 reads: reward, done and trials.

    DyLAN and MacNet returned a 2-tuple here, so `--mas_type dylan` and
    `--mas_type macnet` died on the first completed task with a ValueError.
    """
    env = FakeEnv(max_trials=2, steps_to_done=1)
    workflow = build_workflow(mas_type, env)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert isinstance(result, EpisodeResult), (
        f"{mas_type}.schedule() returned {type(result).__name__}, expected EpisodeResult"
    )
    assert isinstance(result.reward, numbers.Real), f"{mas_type}: reward is {result.reward!r}"
    assert isinstance(result.done, bool), f"{mas_type}: done is {result.done!r}"
    assert isinstance(result.trials, int), f"{mas_type}: trials is {result.trials!r}"


@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_schedule_result_still_unpacks_positionally(mas_type):
    """The NamedTuple keeps `reward, done, trials = schedule(...)` working.

    Analysis notebooks and Slurm wrappers outside this repo unpack the result
    positionally; that is why EpisodeResult is a NamedTuple and not a plain
    dataclass.
    """
    env = FakeEnv(max_trials=2, steps_to_done=1)
    workflow = build_workflow(mas_type, env)

    reward, done, trials = workflow.schedule(
        {"task_main": "m", "task_description": "d", "few_shots": []}
    )

    assert (reward, done, trials) == tuple(
        workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})
    )


@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_schedule_rejects_a_task_config_missing_its_required_keys(mas_type):
    env = FakeEnv()
    workflow = build_workflow(mas_type, env)

    with pytest.raises(ValueError, match="task_main|task_description"):
        workflow.schedule({})


# ── D1 · all four recorders share one signature ───────────────────────────────

@pytest.fixture
def working_dir(tmp_path):
    return str(tmp_path)


def build_recorder(task: str, working_dir: str):
    return RECORDERS[task](working_dir=working_dir, namespace=f"{task}-contract")


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_task_end_accepts_reward_done_and_trials(task, working_dir):
    """run.py:132 calls task_end(reward, done, trials) on every recorder.

    AlfworldRecorder took two arguments, so --task alfworld - the argparse
    default - raised TypeError on the first completed task.
    """
    recorder = build_recorder(task, working_dir)
    recorder.task_begin(0, dict(TASK_CONFIGS[task]))

    recorder.task_end(1.0, True, 7)

    assert recorder.total_rewards == [1.0], (
        f"{task}: task_end did not reach BaseRecorder, so total_rewards stayed {recorder.total_rewards}"
    )
    assert recorder.total_dones == [True]
    assert recorder.total_trials == [7]


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_average_results_returns_three_numerics(task, working_dir):
    recorder = build_recorder(task, working_dir)
    recorder.task_begin(0, dict(TASK_CONFIGS[task]))
    recorder.task_end(1.0, True, 7)

    results = recorder.average_results()

    assert len(results) == 3, (
        f"{task}: average_results() returned {len(results)} values, run.py:143 unpacks 3"
    )
    assert all(isinstance(value, numbers.Real) for value in results), f"{task}: {results!r}"


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_average_results_works_before_any_task_has_ended(task, working_dir):
    """A sweep over an empty task list must report zeros, not divide by zero.

    Both AlfworldRecorder and PDDLRecorder divided by a count that starts at 0.
    """
    recorder = build_recorder(task, working_dir)

    rewards, dones, trials = recorder.average_results()

    assert (rewards, dones, trials) == (0, 0, 0)


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_task_end_before_task_begin_is_rejected(task, working_dir):
    """The base class refuses to record a result it cannot attribute to a task."""
    recorder = build_recorder(task, working_dir)

    with pytest.raises(RuntimeError, match="task id or the task config"):
        recorder.task_end(1.0, True, 7)
