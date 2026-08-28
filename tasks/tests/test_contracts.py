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

from tasks.mas_workflow import MAS
from tasks.tests.fakes import (
    FakeEmbeddingFunc,
    FakeEnv,
    FakeLLM,
    RecordingObserver,
    fake_reasoning,
)

MEMORY_CONFIG_KEYS = ("successful_topk", "failed_topk", "insights_topk", "threshold")


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
