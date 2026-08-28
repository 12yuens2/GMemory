"""The agent retry budget.

Two properties: the budget is spent on every route through the loop, and running
out of it ends one episode rather than the sweep.

Tests exercising the empty-response path use a fake with a call ceiling, so an
unbounded loop fails in milliseconds instead of hanging the suite.
"""

import pytest

from mas.mas import AgentCallFailed, EpisodeResult, MetaMAS, RetryAgentCall

from tasks.tests.fakes import FakeEnv, FakeLLM, RecordingObserver
from tasks.tests.test_contracts import build_workflow

TASK = {"task_main": "m", "task_description": "d", "few_shots": []}


class Harness(MetaMAS):
    """Minimal MetaMAS just to drive _call_agent_with_retries directly."""

    def build_system(self, *args, **kwargs):
        raise NotImplementedError

    def schedule(self, task_config: dict):
        raise NotImplementedError


@pytest.fixture
def harness():
    mas = Harness()
    mas.add_observer(RecordingObserver())
    return mas


# ── the budget is spent on every path ─────────────────────────────────────────

def test_an_empty_response_spends_a_try(harness):
    attempts = []

    with pytest.raises(AgentCallFailed):
        harness._call_agent_with_retries(
            lambda: attempts.append(1) or "", description="solver agent"
        )

    assert len(attempts) == 3, (
        f"an always-empty agent was called {len(attempts)} times, expected the 3-try budget"
    )


def test_a_raised_exception_spends_a_try(harness):
    attempts = []

    def explode():
        attempts.append(1)
        raise ValueError("backend is down")

    with pytest.raises(AgentCallFailed):
        harness._call_agent_with_retries(explode, description="solver agent")

    assert len(attempts) == 3


def test_a_retry_request_spends_a_try(harness):
    attempts = []

    def rejected():
        attempts.append(1)
        raise RetryAgentCall("validator returned INVALID")

    with pytest.raises(AgentCallFailed):
        harness._call_agent_with_retries(rejected, description="solver agent")

    assert len(attempts) == 3


def test_a_late_success_is_returned(harness):
    attempts = []

    def flaky():
        attempts.append(1)
        if len(attempts) < 3:
            raise ConnectionError("transient")
        return "go to desk 1"

    assert harness._call_agent_with_retries(flaky, description="solver agent") == "go to desk 1"
    assert len(attempts) == 3


def test_exhaustion_raises_rather_than_returning_nothing(harness):
    """A caller must never be handed an unusable action, or none at all."""
    with pytest.raises(AgentCallFailed, match="produced no usable action in 3 attempts"):
        harness._call_agent_with_retries(lambda: "", description="solver agent")


def test_the_underlying_error_is_kept_as_the_cause(harness):
    def explode():
        raise ValueError("backend is down")

    with pytest.raises(AgentCallFailed) as raised:
        harness._call_agent_with_retries(explode, description="solver agent")

    assert isinstance(raised.value.__cause__, ValueError)


def test_a_rejected_action_is_used_once_the_budget_runs_out(harness):
    """A reviewer-rejected action is still an action; an empty one is not."""
    calls = []

    def rejected():
        raise RetryAgentCall("nope", fallback=lambda: calls.append(1) or "look at desk 1")

    assert harness._call_agent_with_retries(rejected, description="solver agent") == "look at desk 1"
    assert len(calls) == 1, "the fallback should only be built when it is actually needed"


# ── an always-empty LLM ends the episode, and nothing else ────────────────────

@pytest.mark.parametrize("mas_type", ["autogen", "autogen_mas"])
def test_an_always_empty_llm_does_not_hang(mas_type):
    env = FakeEnv(max_trials=2)
    workflow = build_workflow(mas_type, env, replies=[""])

    result = workflow.schedule(dict(TASK))

    assert result.done is False
    assert env.actions == [], "no action should have reached the environment"


@pytest.mark.parametrize("mas_type", ["autogen", "autogen_mas"])
def test_an_always_failing_llm_does_not_hang(mas_type):
    env = FakeEnv(max_trials=2)
    workflow = build_workflow(mas_type, env)
    workflow.get_agent("solver").reasoning.llm_model = FakeLLM(max_calls=0)

    result = workflow.schedule(dict(TASK))

    assert result.done is False


@pytest.mark.parametrize("mas_type", ["autogen", "autogen_mas"])
def test_a_failed_agent_scores_the_task_rather_than_dropping_it(mas_type):
    """A failed agent ends the episode, which is still scored - the task is not
    dropped from the denominator."""
    env = FakeEnv(max_trials=5)
    workflow = build_workflow(mas_type, env, replies=[""])
    observer = workflow.observers[0]

    result = workflow.schedule(dict(TASK))

    assert result.reward == 0.0
    assert result.done is False
    assert any("Ending episode at trial 1" in message for message in observer.messages), (
        "the log must say why the episode ended"
    )


@pytest.mark.parametrize("mas_type", ["autogen", "autogen_mas"])
def test_an_aborted_episode_reports_no_trial_count(mas_type):
    """How many turns the task needed was never established, so reporting any
    number - the trial reached, or the full budget - would be inventing one.
    Aggregates leave these out of the mean instead."""
    env = FakeEnv(max_trials=5)
    workflow = build_workflow(mas_type, env, replies=[""])

    result = workflow.schedule(dict(TASK))

    assert result.trials is None
    assert result.done is False, "the task is still scored as unsolved"
    assert any(
        "Ending episode at trial 1" in message for message in workflow.observers[0].messages
    ), "the trial it actually reached still has to be recoverable from the log"


@pytest.mark.parametrize("mas_type", ["autogen", "autogen_mas"])
def test_an_agent_that_recovers_mid_episode_still_completes(mas_type):
    """A transient failure should cost attempts, not the episode."""
    env = FakeEnv(max_trials=4, steps_to_done=1)
    workflow = build_workflow(mas_type, env, replies=["", "", "go to desk 1", "VALID"])

    result = workflow.schedule(dict(TASK))

    assert result.done is True
    assert env.actions, "the recovered action should have reached the environment"


# ── a persistently INVALID verdict is bounded, and the episode advances ───────

def test_a_persistent_invalid_verdict_still_advances_the_episode():
    """Re-prompting the solver spends the shared budget; on exhaustion the
    episode proceeds with the rejected action rather than failing."""
    env = FakeEnv(max_trials=1, steps_to_done=1)
    workflow = build_workflow("autogen_mas", env, replies=["go to desk 1", "INVALID: bad format"])

    result = workflow.schedule(dict(TASK))

    assert isinstance(result, EpisodeResult)
    assert env.actions == ["go to desk 1"], (
        f"the episode should have taken the rejected action, took {env.actions}"
    )


def test_a_persistent_invalid_verdict_re_prompts_at_most_three_times():
    env = FakeEnv(max_trials=1, steps_to_done=1)
    workflow = build_workflow("autogen_mas", env, replies=["go to desk 1", "INVALID: bad format"])
    solver_llm = workflow.get_agent("solver").reasoning.llm_model

    workflow.schedule(dict(TASK))

    # The validator prompt is the one carrying "Solver's latest response", so
    # counting those counts attempts regardless of what else the fake serves.
    validations = [
        call for call in solver_llm.calls
        if "Solver's latest response" in call[-1].content
    ]
    assert len(validations) == 3, f"the solver was validated {len(validations)} times, expected 3"
