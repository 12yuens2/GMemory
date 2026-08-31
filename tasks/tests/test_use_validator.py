"""`--use_validator` adds an agent that reviews the solver's action format."""

import pytest

from mas.mas import EpisodeResult
from tasks.mas_workflow import MAS

from tasks.tests.fakes import FakeEnv
from tasks.tests.test_contracts import build_workflow

TASK = {"task_main": "put a mug on the desk", "task_description": "a description", "few_shots": []}


def agent_names(workflow) -> set:
    return set(workflow.agents_team)


# ── the flag decides the team ─────────────────────────────────────────────────

def test_without_the_flag_there_is_no_validator_agent():
    workflow = build_workflow("autogen", FakeEnv(), use_validator=False)

    assert "validator" not in agent_names(workflow), (
        "a validator was hired for a run that did not ask for one"
    )


def test_with_the_flag_the_validator_joins_the_team():
    workflow = build_workflow("autogen", FakeEnv(), use_validator=True)

    assert "validator" in agent_names(workflow), f"team is {sorted(agent_names(workflow))}"


def test_the_validator_carries_the_validator_system_prompt():
    from tasks.mas_workflow.autogen.autogen_prompt import AUTOGEN_PROMPT

    workflow = build_workflow("autogen", FakeEnv(), use_validator=True)

    assert workflow.get_agent("validator").system_instruction == (
        AUTOGEN_PROMPT.validator_system_prompt
    )


# ── the flag decides the memories ─────────────────────────────────────────────

def test_without_the_flag_there_is_one_memory():
    workflow = build_workflow("autogen", FakeEnv(), use_validator=False)

    assert getattr(workflow, "meta_memory_validator", None) is None, (
        "a validator memory was built for a run with no validator"
    )


def test_with_the_flag_the_validator_gets_its_own_memory():
    workflow = build_workflow("autogen", FakeEnv(), use_validator=True)

    assert workflow.meta_memory_validator is not workflow.meta_memory, (
        "the validator shares the solver's memory, so their updates overwrite each other"
    )
    assert workflow.meta_memory_validator.namespace == workflow.meta_memory.namespace + "_validator"


# ── the flag decides how many model calls a trial costs ───────────────────────

def model_calls(workflow) -> int:
    """Every agent shares one reasoning module here, so this counts the trial."""
    return len(workflow.get_agent("solver").reasoning.llm_model.calls)


def test_one_model_call_per_trial_without_the_validator():
    env = FakeEnv(max_trials=1, steps_to_done=1)
    workflow = build_workflow("autogen", env, replies=["go to desk 1"], use_validator=False)

    workflow.schedule(dict(TASK))

    assert model_calls(workflow) == 1, (
        f"the solver should be the only agent called, got {model_calls(workflow)} calls"
    )


def test_two_model_calls_per_trial_with_the_validator():
    env = FakeEnv(max_trials=1, steps_to_done=1)
    workflow = build_workflow(
        "autogen", env, replies=["go to desk 1", "VALID"], use_validator=True
    )

    workflow.schedule(dict(TASK))

    assert model_calls(workflow) == 2, (
        f"the solver proposes and the validator reviews, got {model_calls(workflow)} calls"
    )


# ── either setting still satisfies the workflow contract ─────────────────────

@pytest.mark.parametrize("use_validator", [False, True])
def test_either_setting_returns_an_episode_result(use_validator):
    env = FakeEnv(max_trials=2, steps_to_done=1)
    workflow = build_workflow("autogen", env, use_validator=use_validator)

    result = workflow.schedule(dict(TASK))

    assert isinstance(result, EpisodeResult)
    assert isinstance(result.trials, int)


@pytest.mark.parametrize("use_validator", [False, True])
def test_either_setting_saves_the_task_context(use_validator):
    env = FakeEnv(max_trials=1, steps_to_done=1)
    workflow = build_workflow("autogen", env, use_validator=use_validator)

    workflow.schedule(dict(TASK))

    assert workflow.meta_memory.current_task_context is not None


# ── the registry no longer carries a second copy ──────────────────────────────

def test_the_workflow_registry_has_no_separate_validator_entry():
    """`autogen_mas` was a second class of the same name, aliased on import."""
    assert "autogen_mas" not in MAS, (
        "the validator is a flag now, so it must not also be a mas_type"
    )
