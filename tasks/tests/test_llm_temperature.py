"""The sampling temperature a caller sets reaches the endpoint.

A configured-but-unsent sampling parameter is worse than no parameter at all: it
implies control over decoding that is not exercised, and it makes any determinism
claim attached to a seed untrue.

Not every OpenAI-compatible endpoint accepts the parameter, so the second half of
this module pins the fallback: a refusal costs the call one extra request, not the
run.
"""

import pytest

from mas.llm import GPTChat, LLMCallFailed, Message
from mas.reasoning import ReasoningConfig, ReasoningIO
from mas.settings import LLMSettings

from tasks.tests.fakes import (
    TemperatureRejectingCompletions,
    chat_over_fake_completions,
)

PROMPT = [Message("user", "what is the next action?")]


def settings(temperature: float) -> LLMSettings:
    return LLMSettings(
        api_base="http://localhost:9999", api_key="test-key", max_tokens=64,
        temperature=temperature,
    )


# ── the value reaches the request ─────────────────────────────────────────────

def test_the_temperature_a_caller_sets_reaches_the_request():
    chat, completions = chat_over_fake_completions(["go to desk 1"])

    chat(PROMPT, temperature=0.7)

    assert completions.calls[0]["temperature"] == 0.7, (
        f"temperature was not sent; request was {sorted(completions.calls[0])}"
    )


def test_temperature_zero_is_sent_rather_than_treated_as_unset():
    """Every workflow asks for greedy decoding, and 0.0 is falsy."""
    chat, completions = chat_over_fake_completions(["go to desk 1"], settings=settings(0.9))

    chat(PROMPT, temperature=0)

    assert completions.calls[0]["temperature"] == 0, (
        "0 was replaced by the configured default, so no workflow can ask for greedy decoding"
    )


def test_a_caller_that_sets_nothing_gets_the_configured_default():
    """Memory modules call the LLM with no sampling arguments at all."""
    chat, completions = chat_over_fake_completions(["a summary"], settings=settings(0.35))

    chat(PROMPT)

    assert completions.calls[0]["temperature"] == 0.35


def test_the_reasoning_config_temperature_reaches_the_request():
    """The path every workflow actually uses: ReasoningConfig -> ReasoningIO -> GPTChat."""
    chat, completions = chat_over_fake_completions(["go to desk 1"], settings=settings(0.9))
    reasoning = ReasoningIO(llm_model=chat)

    reasoning(PROMPT, ReasoningConfig(temperature=0, stop_strs=["\n"]))

    assert completions.calls[0]["temperature"] == 0


# ── an endpoint that refuses the parameter ────────────────────────────────────

def test_an_endpoint_that_refuses_temperature_still_answers():
    refusing = TemperatureRejectingCompletions(["go to desk 1"])
    chat, completions = chat_over_fake_completions(None, completions=refusing)

    answer = chat(PROMPT, temperature=0)

    assert answer == "go to desk 1", "a refused parameter cost the call its answer"
    assert refusing.refusals == 1
    assert "temperature" not in completions.calls[-1], (
        "the retry offered temperature again"
    )


def test_the_refusal_is_remembered_so_it_costs_one_request_not_one_per_call():
    refusing = TemperatureRejectingCompletions(["go to desk 1"])
    chat, _ = chat_over_fake_completions(None, completions=refusing)

    for _ in range(4):
        chat(PROMPT, temperature=0)

    assert refusing.refusals == 1, (
        f"the endpoint was offered temperature {refusing.refusals} times; a refusal "
        f"should be remembered for the life of the client"
    )
    assert len(refusing.calls) == 5, "4 answers plus the single refused attempt"


def test_an_error_that_is_not_about_temperature_is_not_swallowed():
    """The fallback must not turn an unrelated 400 into a silent retry."""
    chat, _ = chat_over_fake_completions([ValueError("model `nonesuch` does not exist")])

    with pytest.raises(LLMCallFailed):
        chat(PROMPT, temperature=0)


def test_a_refusal_does_not_spend_the_retry_budget():
    """Dropping an unsupported parameter is not a failed attempt."""
    refusing = TemperatureRejectingCompletions([None, None, None, None, "answered at last"])
    chat, _ = chat_over_fake_completions(None, completions=refusing)

    answer = chat(PROMPT, temperature=0)

    assert answer == "answered at last", (
        "the refused attempt consumed one of the five retries"
    )


def test_the_settings_default_is_used_when_the_reasoning_config_leaves_it_unset():
    """ReasoningConfig.temperature defaults to None, which must mean 'as configured'."""
    chat, completions = chat_over_fake_completions(["go to desk 1"], settings=settings(0.2))
    reasoning = ReasoningIO(llm_model=chat)

    reasoning(PROMPT, ReasoningConfig())

    assert completions.calls[0]["temperature"] == 0.2


# ── the parameter is declared where it is honoured ────────────────────────────

def test_gptchat_does_not_claim_the_parameter_is_ignored():
    """The docstring on LLMSettings.temperature is part of the contract."""
    from mas.settings import LLMSettings as Settings

    assert "not sent to the API" not in (Settings.__doc__ or ""), (
        "LLMSettings still documents temperature as unsent"
    )


def test_no_call_site_leaves_temperature_commented_out():
    import inspect

    source = inspect.getsource(GPTChat.__call__)
    assert "#temperature" not in source.replace(" ", ""), (
        "temperature is still commented out at the call site"
    )
