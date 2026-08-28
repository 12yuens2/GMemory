"""Contract tests across the registries.

Each family - workflows, recorders, memory modules - has one contract its callers
rely on, asserted here once over the whole family rather than per implementation.

These read the registries (`MAS`, `RECORDERS`, `ENVS`) rather than hard-coded
lists, so a new implementation is covered the moment it is registered.
"""

import numbers
import re
import tempfile

import pytest

from mas.mas import EpisodeResult
from mas.memory import MASMemoryBase
from mas.module_map import module_map

from tasks.envs import RECORDERS
from tasks.envs.base_env import AggregateResults

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
    """schedule() must return what run.py reads: reward, done and trials."""
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
    """`reward, done, trials = schedule(...)` must keep working.

    Call sites outside this repo unpack the result positionally.
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
    """run.py hands every recorder the EpisodeResult a workflow returned."""
    recorder = build_recorder(task, working_dir)
    recorder.task_begin(0, dict(TASK_CONFIGS[task]))

    recorder.task_end(EpisodeResult(reward=1.0, done=True, trials=7))

    assert recorder.total_rewards == [1.0], (
        f"{task}: task_end did not reach BaseRecorder, so total_rewards stayed {recorder.total_rewards}"
    )
    assert recorder.total_dones == [True]
    assert recorder.total_trials == [7]


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_average_results_returns_three_numerics(task, working_dir):
    recorder = build_recorder(task, working_dir)
    recorder.task_begin(0, dict(TASK_CONFIGS[task]))
    recorder.task_end(EpisodeResult(reward=1.0, done=True, trials=7))

    results = recorder.average_results()

    assert isinstance(results, AggregateResults), (
        f"{task}: average_results() returned {type(results).__name__}"
    )
    assert all(isinstance(value, numbers.Real) for value in results), f"{task}: {results!r}"


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_average_results_works_before_any_task_has_ended(task, working_dir):
    """A sweep over an empty task list must report zeros, not divide by zero."""
    recorder = build_recorder(task, working_dir)

    assert recorder.average_results() == (0, 0, 0)


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_task_end_before_task_begin_is_rejected(task, working_dir):
    """The base class refuses to record a result it cannot attribute to a task."""
    recorder = build_recorder(task, working_dir)

    with pytest.raises(RuntimeError, match="task id or the task config"):
        recorder.task_end(EpisodeResult(reward=1.0, done=True, trials=7))


# ── D4 · every memory module works with every workflow ────────────────────────

# Read from module_map so a newly registered module is covered without editing
# this list. 'empty' is MASMemoryBase itself.
MEMORY_KEYS = [
    "empty",
    "voyager",
    "memorybank",
    "chatdev",
    "generative",
    "metagpt",
    "intrinsicmemory-pddl",
    "intrinsicmemory-fever",
    "intrinsicmemory-alfworld",
    "intrinsicmemory-llm-structured-template",
    "intrinsicmemory-notemplate",
]


# g-memory persists through langchain_chroma, which this suite stubs, so it cannot
# be driven offline. Named here so the omission is explicit.
UNTESTABLE_OFFLINE = {"g-memory"}


def test_the_memory_matrix_covers_every_registered_module():
    """Fails when a module is added to module_map but not to MEMORY_KEYS.

    module_map exposes no registry, but its error message lists every allowed
    value.
    """
    with pytest.raises(ValueError, match="Allowed values") as raised:
        module_map("io", "not-a-memory-module")
    registered = set(re.findall(r"'([^']+)'", str(raised.value))) - {"not-a-memory-module"}

    assert registered == set(MEMORY_KEYS) | UNTESTABLE_OFFLINE, (
        f"module_map and MEMORY_KEYS disagree: {registered ^ (set(MEMORY_KEYS) | UNTESTABLE_OFFLINE)}"
    )


@pytest.mark.parametrize("mas_type", sorted(MAS))
@pytest.mark.parametrize("memory_key", MEMORY_KEYS)
def test_every_memory_module_survives_every_workflow(mas_type, memory_key):
    """Every memory module must survive an episode driven by any workflow.

    Mostly this exercises the summarize() keyword contract, which is the one
    place the workflows and the memory modules have to agree.
    """
    _, memory_cls = module_map("io", memory_key)
    env = FakeEnv(max_trials=2, steps_to_done=1)
    workflow = build_workflow(mas_type, env, memory_cls=memory_cls)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert isinstance(result, EpisodeResult)


# ── the trials convention ─────────────────────────────────────────────────────

@pytest.mark.parametrize("mas_type", sorted(MAS))
@pytest.mark.parametrize("steps_to_done, expected_trials", [(1, 1), (2, 2), (3, 3)])
def test_trials_counts_the_trials_completed(mas_type, steps_to_done, expected_trials):
    """trials is a count of completed trials, not a loop index."""
    env = FakeEnv(max_trials=5, steps_to_done=steps_to_done)
    workflow = build_workflow(mas_type, env)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert result.done is True
    assert result.trials == expected_trials
    assert result.trials == len(env.actions), "trials should match the steps actually taken"


@pytest.mark.parametrize("mas_type", sorted(MAS))
def test_an_unsolved_episode_reports_its_whole_budget(mas_type):
    env = FakeEnv(max_trials=4, steps_to_done=99)
    workflow = build_workflow(mas_type, env)

    result = workflow.schedule({"task_main": "m", "task_description": "d", "few_shots": []})

    assert result.done is False
    assert result.trials == 4
