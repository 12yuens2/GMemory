"""GPTChat's failure and retry behaviour.

The property under test is that an exhausted retry budget is distinguishable from
a model that answered with nothing.

Everything here drives a fake OpenAI client, with time.sleep patched out: the
subject is the retry accounting, not the delay.
"""

from unittest.mock import patch

import pytest

from mas.llm import LLMCallFailed, Message, TokenTracker

from tasks.tests.fakes import StarvedReasoningCompletions
from tasks.tests.fakes import chat_over_fake_completions as build_chat

PROMPT = [Message("system", "you are a solver"), Message("user", "what next?")]


def call_settings(**overrides):
    from mas.settings import LLMSettings

    return LLMSettings(**{
        'api_base': 'http://localhost:9999/v1', 'api_key': 'none', 'max_tokens': 512,
        'max_tokens_ceiling': 8192, 'temperature': 0.1, 'request_timeout': 300.0,
        'log_responses': False, **overrides,
    })


def budgets(completions) -> list:
    """The `max_completion_tokens` of every request, in order."""
    return [call.get('max_completion_tokens') for call in completions.calls]


@pytest.fixture(autouse=True)
def no_sleeping():
    with patch("time.sleep"):
        yield


# ── exhausted retries are distinguishable from an empty answer ───────────────

def test_a_persistent_api_error_raises_rather_than_returning_empty_string():
    chat, _ = build_chat([RuntimeError("500 internal server error")])

    with pytest.raises(LLMCallFailed, match="returned no answer"):
        chat(PROMPT)


def test_the_failure_counts_the_attempts_made_not_the_budget_for_them():
    """A 500 breaks out without retrying, and used to report five attempts anyway.

    Two tests in this file disagreed about the same run: one that the call was
    made once, one that the message said five.
    """
    chat, completions = build_chat([RuntimeError("500 internal server error")])

    with pytest.raises(LLMCallFailed, match="after 1 attempts"):
        chat(PROMPT)

    assert len(completions.calls) == 1


def test_the_underlying_api_error_is_kept_as_the_cause():
    original = RuntimeError("500 internal server error")
    chat, _ = build_chat([original])

    with pytest.raises(LLMCallFailed) as raised:
        chat(PROMPT)

    assert raised.value.__cause__ is original


def test_a_model_that_answers_with_nothing_is_not_an_error():
    """An empty string is a real answer; the caller decides if it is usable."""
    chat, _ = build_chat([""])

    assert chat(PROMPT) == ""


def test_a_none_content_response_is_retried_then_fails():
    chat, completions = build_chat([None])

    with pytest.raises(LLMCallFailed):
        chat(PROMPT)

    assert len(completions.calls) == 5, "a None answer should spend the retry budget"


# ── retry accounting ──────────────────────────────────────────────────────────

def test_a_non_rate_limit_error_is_not_retried():
    """A 500 is not going to be fixed by asking again immediately."""
    chat, completions = build_chat([RuntimeError("500 internal server error")])

    with pytest.raises(LLMCallFailed):
        chat(PROMPT)

    assert len(completions.calls) == 1


def test_a_rate_limit_error_is_retried_with_a_growing_wait():
    chat, completions = build_chat([RuntimeError("429 rate limit exceeded")])

    with patch("time.sleep") as sleep:
        with pytest.raises(LLMCallFailed):
            chat(PROMPT)

    assert len(completions.calls) == 5
    waits = [call.args[0] for call in sleep.call_args_list]
    assert waits == [1, 2, 4, 8, 16], f"waited {waits}, expected a doubling backoff"


def test_a_rate_limit_that_clears_returns_the_answer():
    chat, completions = build_chat([RuntimeError("429 rate limit"), "go to desk 1"])

    assert chat(PROMPT) == "go to desk 1"
    assert len(completions.calls) == 2


# ── token accounting ──────────────────────────────────────────────────────────

def test_usage_is_recorded_on_the_shared_tracker():
    tracker = TokenTracker()
    chat, _ = build_chat(["go to desk 1"], tracker=tracker)

    chat(PROMPT)

    assert (tracker.prompt_tokens, tracker.completion_tokens) == (11, 7)
    assert (tracker.intrinsic_prompt_tokens, tracker.intrinsic_completion_tokens) == (0, 0)


def test_an_intrinsic_call_is_billed_to_the_intrinsic_columns_too():
    """intrinsic=True is the only route to the intrinsic_* columns."""
    tracker = TokenTracker()
    chat, _ = build_chat(["updated memory"], tracker=tracker)

    chat(PROMPT, intrinsic=True)

    assert tracker.intrinsic_prompt_tokens == 11
    assert tracker.intrinsic_completion_tokens == 7
    assert tracker.intrinsic_prompt_tokens <= tracker.prompt_tokens


# ── a request cannot take forever ─────────────────────────────────────────────

def test_the_client_is_built_with_the_configured_request_timeout():
    """Nothing above GPTChat can bound this: it builds its own client."""
    from mas.llm import GPTChat
    from mas.settings import LLMSettings

    settings = LLMSettings(
        api_base="http://localhost:9999/v1", api_key="none", max_tokens=512,
        max_tokens_ceiling=8192, temperature=0.1, request_timeout=42.0, log_responses=False,
    )

    chat = GPTChat(model_name="fake-model", settings=settings)

    assert chat.client.timeout == 42.0, (
        "a wedged endpoint consumes the allocation instead of failing the job"
    )


# ── a starved reasoning model is retried with more budget, not the same one ────

def test_a_starved_reasoning_model_is_retried_with_a_larger_budget():
    """Asking again for the same ceiling gets the same truncation.

    A reasoning model spends `max_completion_tokens` on reasoning and returns
    content=None. The budget is the only thing whose change can answer the call,
    so it is what the retry changes.
    """
    completions = StarvedReasoningCompletions(["the answer"], needs=2000)
    chat, _ = build_chat(None, settings=call_settings(), completions=completions)

    assert chat(PROMPT) == "the answer"
    assert budgets(completions) == [512, 1024, 2048], budgets(completions)


def test_no_two_attempts_of_a_starved_call_ask_for_the_same_budget():
    """The property the old loop broke: five identical requests, five identical
    truncations, and five budgets of reasoning paid for."""
    completions = StarvedReasoningCompletions(["unreachable"], needs=10 ** 9)
    chat, _ = build_chat(None, settings=call_settings(), completions=completions)

    with pytest.raises(LLMCallFailed):
        chat(PROMPT)

    asked = budgets(completions)
    assert len(asked) == len(set(asked)), f'repeated an identical request: {asked}'


def test_the_budget_never_exceeds_the_ceiling_the_run_set():
    completions = StarvedReasoningCompletions(["unreachable"], needs=10 ** 9)
    chat, _ = build_chat(
        None, settings=call_settings(max_tokens=512, max_tokens_ceiling=2048),
        completions=completions,
    )

    with pytest.raises(LLMCallFailed):
        chat(PROMPT)

    assert budgets(completions) == [512, 1024, 2048], budgets(completions)


def test_a_starved_call_with_no_headroom_is_not_retried_at_all():
    """Where the budget cannot grow, another attempt is the same request again.

    This is the waste the fix removes: at a 4096 ceiling five attempts cost
    20,480 completion tokens to fail as deterministically as one does.
    """
    completions = StarvedReasoningCompletions(["unreachable"], needs=10 ** 9)
    chat, _ = build_chat(
        None, settings=call_settings(max_tokens=4096, max_tokens_ceiling=4096),
        completions=completions,
    )

    with pytest.raises(LLMCallFailed):
        chat(PROMPT)

    assert budgets(completions) == [4096], budgets(completions)


def test_a_ceiling_below_the_budget_asked_for_does_not_shrink_it():
    """The ceiling bounds where a retry may climb to, and nothing else."""
    completions = StarvedReasoningCompletions(["unreachable"], needs=10 ** 9)
    chat, _ = build_chat(
        None, settings=call_settings(max_tokens=4096, max_tokens_ceiling=512),
        completions=completions,
    )

    with pytest.raises(LLMCallFailed):
        chat(PROMPT)

    assert budgets(completions) == [4096], budgets(completions)


def test_the_failure_names_the_budget_it_could_not_answer_within():
    """`after 5 attempts` said nothing about which of the causes ran out."""
    completions = StarvedReasoningCompletions(["unreachable"], needs=10 ** 9)
    chat, _ = build_chat(
        None, settings=call_settings(max_tokens=512, max_tokens_ceiling=2048),
        completions=completions,
    )

    with pytest.raises(LLMCallFailed, match="max_completion_tokens=2048") as raised:
        chat(PROMPT)

    assert "3 attempts" in str(raised.value), raised.value


def test_a_starved_attempt_followed_by_an_api_error_does_not_blame_the_budget():
    """Whatever ended the call is what the message names.

    A budget mentioned for a failure the budget did not cause is the log
    asserting a cause the endpoint never reported.
    """
    class StarvedThenBroken(StarvedReasoningCompletions):
        def create(self, **kwargs):
            if self.calls:
                raise RuntimeError("500 internal server error")
            return super().create(**kwargs)

    completions = StarvedThenBroken(["unreachable"], needs=10 ** 9)
    chat, _ = build_chat(None, settings=call_settings(), completions=completions)

    with pytest.raises(LLMCallFailed) as raised:
        chat(PROMPT)

    assert 'max_completion_tokens' not in str(raised.value), raised.value


def test_the_budget_spent_on_a_starved_attempt_is_still_billed():
    """Every attempt generated a full budget of reasoning, and the sweep sizes
    its jobs off these counts."""
    tracker = TokenTracker()
    completions = StarvedReasoningCompletions(["the answer"], needs=2000)
    chat, _ = build_chat(
        None, tracker=tracker, settings=call_settings(), completions=completions,
    )

    chat(PROMPT)

    assert tracker.completion_tokens == 512 + 1024 + completions.completion_tokens, (
        f'billed {tracker.completion_tokens}'
    )


def test_a_none_answer_with_no_reasoning_still_spends_the_retry_budget():
    """Only the starved cause is deterministic. An endpoint that sent neither
    content nor reasoning is unexplained, and may yet answer."""
    chat, completions = build_chat([None], settings=call_settings())

    with pytest.raises(LLMCallFailed) as raised:
        chat(PROMPT)

    assert budgets(completions) == [512] * 5, budgets(completions)
    assert 'max_completion_tokens' not in str(raised.value), raised.value
