"""GPTChat's failure and retry behaviour.

The property under test is that an exhausted retry budget is distinguishable from
a model that answered with nothing.

Everything here drives a fake OpenAI client, with time.sleep patched out: the
subject is the retry accounting, not the delay.
"""

from types import SimpleNamespace
from unittest.mock import patch

import pytest

from mas.llm import GPTChat, LLMCallFailed, Message, TokenTracker

PROMPT = [Message("system", "you are a solver"), Message("user", "what next?")]


class FakeCompletions:
    """Stands in for client.chat.completions, scripted per call.

    Each entry in `script` is either an answer string, None (the model returned
    no content), or an exception instance to raise.
    """

    def __init__(self, script):
        self.script = list(script)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        outcome = self.script[min(len(self.calls) - 1, len(self.script) - 1)]
        if isinstance(outcome, BaseException):
            raise outcome
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=outcome))],
            usage=SimpleNamespace(prompt_tokens=11, completion_tokens=7),
        )


def build_chat(script, tracker=None) -> tuple[GPTChat, FakeCompletions]:
    chat = GPTChat(model_name="fake-model", tracker=tracker)
    completions = FakeCompletions(script)
    chat.client = SimpleNamespace(chat=SimpleNamespace(completions=completions))
    return chat, completions


@pytest.fixture(autouse=True)
def no_sleeping():
    with patch("time.sleep"):
        yield


# ── exhausted retries are distinguishable from an empty answer ───────────────

def test_a_persistent_api_error_raises_rather_than_returning_empty_string():
    chat, _ = build_chat([RuntimeError("500 internal server error")])

    with pytest.raises(LLMCallFailed, match="returned no answer after 5 attempts"):
        chat(PROMPT)


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
