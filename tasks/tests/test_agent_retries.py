"""The agent retry budget (test plan items B2, B3, E3).

Each workflow used to carry its own copy of

    tries = 0
    while tries < 3:
        try:
            action = agent.response(...)
            if action == '':
                continue          # <- jumps past the increment below
            action = env.process_action(action)
            break
        except Exception as e:
            print(...)
        tries += 1

The `continue` skipped `tries += 1`, so an agent that kept returning "" spun
without limit - and GPTChat returned "" on any non-rate-limit API error, which is
the failure most likely to persist. The parent process waited on
`future.result()` with no timeout, so one bad endpoint hung a whole sweep.

Every test here that exercises the empty-response path is written against a fake
with a call ceiling, so the unbounded version fails in milliseconds rather than
hanging the suite.
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


# ── the budget is spent on every path, not just the exception path ─────────────

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


def test_exhaustion_raises_rather_than_leaving_the_action_unbound(harness):
    """The caller used to fall through to AgentMessage(message=action) with
    `action` never bound, so the real failure surfaced as a NameError."""
    with pytest.raises(AgentCallFailed, match="produced no usable action in 3 attempts"):
        harness._call_agent_with_retries(lambda: "", description="solver agent")


def test_the_underlying_error_is_kept_as_the_cause(harness):
    def explode():
        raise ValueError("backend is down")

    with pytest.raises(AgentCallFailed) as raised:
        harness._call_agent_with_retries(explode, description="solver agent")

    assert isinstance(raised.value.__cause__, ValueError)


def test_a_rejected_action_is_used_once_the_budget_runs_out(harness):
    """A reviewer-rejected action is still an action; only an empty one is not."""
    calls = []

    def rejected():
        raise RetryAgentCall("nope", fallback=lambda: calls.append(1) or "look at desk 1")

    assert harness._call_agent_with_retries(rejected, description="solver agent") == "look at desk 1"
    assert len(calls) == 1, "the fallback should only be built when it is actually needed"


# ── B2 · an always-empty LLM ends the episode, and nothing else ───────────────

@pytest.mark.parametrize("mas_type", ["autogen", "autogen_mas"])
def test_an_always_empty_llm_does_not_hang(mas_type):
    """On the previous code this looped without limit; the call ceiling on
    FakeLLM turns that into a fast failure instead of a hung suite."""
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
    """The failure has to stay inside one episode.

    AgentCallFailed used to propagate out of schedule, through run_task's loop
    over tasks, to run_experiment's except - abandoning every remaining task in
    the dataset over one bad LLM call. It now ends the episode, which is still
    scored, with an accurate trial count.
    """
    env = FakeEnv(max_trials=5)
    workflow = build_workflow(mas_type, env, replies=[""])
    observer = workflow.observers[0]

    result = workflow.schedule(dict(TASK))

    assert result.reward == 0.0
    assert result.done is False
    assert result.trials == 0, "the episode ended on its first trial"
    assert any("Ending episode at trial 1" in message for message in observer.messages), (
        "the log must say why the episode ended"
    )


@pytest.mark.parametrize("mas_type", ["autogen", "autogen_mas"])
def test_an_agent_that_recovers_mid_episode_still_completes(mas_type):
    """A transient failure should cost attempts, not the episode."""
    env = FakeEnv(max_trials=4, steps_to_done=1)
    workflow = build_workflow(mas_type, env, replies=["", "", "go to desk 1", "VALID"])

    result = workflow.schedule(dict(TASK))

    assert result.done is True
    assert env.actions, "the recovered action should have reached the environment"


# ── B3 · a persistently INVALID verdict is bounded and the episode advances ────

def test_a_persistent_invalid_verdict_still_advances_the_episode():
    """AutoGenMAS re-prompts the solver whenever the validator says INVALID.

    That path had no bound of its own; it now spends the shared budget, and on
    exhaustion the episode proceeds with the rejected action rather than dying,
    because refusing to act at all is worse than acting on a disputed answer.
    """
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
    # counting those counts attempts without depending on how many other calls
    # the shared fake also serves.
    validations = [
        call for call in solver_llm.calls
        if "Solver's latest response" in call[-1].content
    ]
    assert len(validations) == 3, f"the solver was validated {len(validations)} times, expected 3"
