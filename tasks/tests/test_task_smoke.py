"""One episode per (mas_type x task), and the env interface contract.

Phase 1 found six blocking defects. Every one of them sat on a path that no test
reached, because the 115 tests then in the repository exercised one workflow file
against one task's worth of stubs. This module walks the matrix instead: for each
of the four workflows and each of the four tasks it runs the same sequence
run_task does -

    recorder.dataset_begin()
    recorder.task_begin(...)
    mas.schedule(task_config)
    recorder.task_end(reward, done, trials)
    recorder.average_results()
    recorder.dataset_end()

- with the real recorder, the real prompt module and the real tasks/configs.yaml,
and a fake environment and LLM in place of a simulator and an endpoint. That
combination is what would have caught the AlfworldRecorder arity bug, the two
workflows returning the wrong arity, and the MacNet summarize keyword, all in one
run.

The environments themselves cannot be constructed offline - ALFWorld's is
commented out, ScienceWorld needs a server, PDDL's needs pddlgym - so their side
of the contract is asserted against the classes rather than instances.
"""

import inspect

import pytest
import yaml

from mas.mas import EpisodeResult

from tasks.envs import ENVS, RECORDERS
from tasks.envs.base_env import BaseEnv, BaseRecorder
from tasks.mas_workflow import MAS
from tasks.prompts import get_dataset_system_prompt, get_task_few_shots
from tasks.tests.fakes import FakeEnv
from tasks.tests.test_contracts import TASK_CONFIGS, build_workflow

with open("tasks/configs.yaml") as reader:
    TASK_SETTINGS: dict = yaml.safe_load(reader)

# What each task's prompt and recorder code reads out of a task config. The
# fixtures in test_contracts carry the recorder's keys; these add the prompts'.
SMOKE_TASK_CONFIGS = {
    "alfworld": {**TASK_CONFIGS["alfworld"], "task_type": "put"},
    "fever": {**TASK_CONFIGS["fever"], "task": "Ada Lovelace wrote the first program."},
    "pddl": {**TASK_CONFIGS["pddl"]},
    "sciworld": {**TASK_CONFIGS["sciworld"]},
}


# ── D3 · all four envs share one signature ────────────────────────────────────

@pytest.mark.parametrize("task", sorted(ENVS))
def test_every_env_is_a_base_env(task):
    assert issubclass(ENVS[task], BaseEnv)


@pytest.mark.parametrize("task", sorted(ENVS))
@pytest.mark.parametrize(
    "method, expected_returns",
    [
        ("set_env", "tuple[str, str]"),
        ("step", "tuple[str, float, bool]"),
        ("feedback", "tuple[float, bool, str]"),
    ],
)
def test_every_env_declares_the_return_type_its_callers_unpack(task, method, expected_returns):
    """The workflows unpack these positionally, so the arity is the contract.

    Annotations are what is checked here rather than behaviour, because none of
    the four environments can be constructed without its simulator. An annotation
    is a weaker guarantee than a call, but it is the one that was wrong on
    schedule() - both AutoGen workflows returned three values and annotated two -
    so keeping the annotations honest has caught real defects here.
    """
    signature = inspect.signature(getattr(ENVS[task], method))

    assert str(signature.return_annotation).replace("typing.", "") == expected_returns, (
        f"{task}.{method} is annotated {signature.return_annotation}, expected {expected_returns}"
    )


@pytest.mark.parametrize("task", sorted(ENVS))
def test_every_env_accepts_a_config_and_a_trial_budget(task):
    parameters = list(inspect.signature(ENVS[task].__init__).parameters)

    assert parameters[:3] == ["self", "env_config", "max_trials"], (
        f"{task} takes {parameters[:3]}; get_env calls ENVS[task](env_config, max_trials)"
    )


@pytest.mark.parametrize("task", sorted(ENVS))
def test_process_action_does_not_need_an_instance(task):
    """The workflows call env.process_action, but BaseEnv declares it a
    classmethod and two of the four implement it as a staticmethod."""
    assert callable(ENVS[task].process_action)


@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_every_recorder_is_a_base_recorder(task):
    assert issubclass(RECORDERS[task], BaseRecorder)


# ── one episode per (mas_type x task) ─────────────────────────────────────────

@pytest.mark.parametrize("mas_type", sorted(MAS))
@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_one_episode_runs_end_to_end(mas_type, task, tmp_path):
    """The run_task sequence, with the real recorder, prompts and task config."""
    task_config = dict(SMOKE_TASK_CONFIGS[task])
    max_trials = TASK_SETTINGS[task]["max_steps"]

    recorder = RECORDERS[task](working_dir=str(tmp_path), namespace=f"{task}-smoke")
    env = FakeEnv(max_trials=max_trials, steps_to_done=2)
    workflow = build_workflow(mas_type, env)
    workflow.add_observer(recorder)

    recorder.dataset_begin()
    recorder.task_begin(0, task_config)

    task_main, task_description = env.set_env(task_config)
    few_shots = get_task_few_shots(
        dataset=task,
        task_config=task_config,
        few_shots_num=TASK_SETTINGS[task].get("few_shots_num", 0),
    )
    task_config.update(
        task_main=task_main, task_description=task_description, few_shots=few_shots
    )

    task_instruction = get_dataset_system_prompt(task, task_config=task_config)
    for agent in workflow.agents_team.values():
        agent.add_task_instruction(task_instruction)

    episode: EpisodeResult = workflow.schedule(task_config)
    recorder.task_end(episode.reward, episode.done, episode.trials)

    rewards, dones, trials = recorder.average_results()
    recorder.dataset_end()

    assert episode.done is True, f"[{mas_type}/{task}] the episode did not complete"
    assert env.actions, f"[{mas_type}/{task}] no action reached the environment"
    assert (rewards, dones) == (1.0, 1.0), f"[{mas_type}/{task}] recorded {rewards}, {dones}"
    assert trials == episode.trials


@pytest.mark.parametrize("mas_type", sorted(MAS))
@pytest.mark.parametrize("task", sorted(RECORDERS))
def test_an_episode_that_never_solves_still_records_a_result(mas_type, task, tmp_path):
    """The trial budget has to be a bound, not a hope. A task that never reports
    done must still leave a row for the CSV rather than hanging or raising."""
    task_config = dict(SMOKE_TASK_CONFIGS[task])
    recorder = RECORDERS[task](working_dir=str(tmp_path), namespace=f"{task}-unsolved")
    env = FakeEnv(max_trials=3, steps_to_done=99)
    workflow = build_workflow(mas_type, env)
    workflow.add_observer(recorder)

    recorder.task_begin(0, task_config)
    task_main, task_description = env.set_env(task_config)
    task_config.update(task_main=task_main, task_description=task_description, few_shots=[])

    episode = workflow.schedule(task_config)
    recorder.task_end(episode.reward, episode.done, episode.trials)

    assert episode.done is False
    assert len(env.actions) == 3, f"[{mas_type}/{task}] took {len(env.actions)} of 3 trials"
    assert recorder.average_results()[:2] == (0.0, 0)
