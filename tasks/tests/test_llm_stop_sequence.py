"""A stop sequence must not swallow a reasoning model's answer.

A reasoning model streams its hidden reasoning before its answer, in one
continuous generation. An API-level stop sequence is checked against that raw
stream, so a sequence meant to end the answer at its first line-break can fire
inside the reasoning instead, leaving `content` empty. The fallback here mirrors
the temperature one: absorbed once per endpoint, not once per call.
"""

import pytest

from mas.llm import LLMCallFailed, Message
from mas.reasoning import ReasoningConfig, ReasoningIO

from types import SimpleNamespace

from tasks.tests.fakes import (
    FakeCompletions,
    StopTruncatesReasoningCompletions,
    chat_over_fake_completions,
)

PROMPT = [Message("user", "what is the next action?")]


def test_a_stop_sequence_that_truncates_reasoning_is_dropped_and_retried():
    truncating = StopTruncatesReasoningCompletions(["go to desk 1"])
    chat, completions = chat_over_fake_completions(None, completions=truncating)
    reasoning = ReasoningIO(llm_model=chat)

    answer = reasoning(PROMPT, ReasoningConfig(stop_strs=["\n"]))

    assert answer == "go to desk 1", "a truncated reasoning pass cost the call its answer"
    assert truncating.truncations == 1
    assert completions.calls[-1]["stop"] is None, "the retry offered the stop sequence again"


def test_the_drop_is_remembered_so_it_costs_one_request_not_one_per_call():
    truncating = StopTruncatesReasoningCompletions(["go to desk 1"])
    chat, _ = chat_over_fake_completions(None, completions=truncating)
    reasoning = ReasoningIO(llm_model=chat)

    for _ in range(4):
        reasoning(PROMPT, ReasoningConfig(stop_strs=["\n"]))

    assert truncating.truncations == 1, (
        f"the stop sequence was sent {truncating.truncations} times; a truncation should "
        f"be remembered for the life of the client"
    )


class BlanksWithoutReasoning(FakeCompletions):
    """An endpoint that answers `''` while a stop sequence is sent, and reports no
    reasoning at all.

    This is ollama's shape, measured: `qwen3:0.6b` under `stop=['\\n']` returns
    content `''` with no `reasoning_content` field, where gpt-oss on vLLM returns
    content None with the text in `reasoning_content`. Same truncation, so neither
    the empty-versus-None shape nor the presence of a reasoning field can be what
    the fallback turns on.
    """

    def create(self, **kwargs):
        if kwargs.get("stop"):
            self.calls.append(kwargs)
            return SimpleNamespace(
                choices=[SimpleNamespace(message=SimpleNamespace(content=""))],
                usage=SimpleNamespace(
                    prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens
                ),
            )
        return super().create(**kwargs)


def test_an_empty_answer_with_no_reasoning_reported_is_still_the_stop_sequence():
    """The shape ollama sends. Nothing distinguishes it but the stop sequence."""
    chat, completions = chat_over_fake_completions(
        None, completions=BlanksWithoutReasoning(["go to desk 1"])
    )

    assert chat(PROMPT, stop_strs=["\n"]) == "go to desk 1"
    assert completions.calls[-1]["stop"] is None


def test_an_empty_answer_is_still_the_callers_to_judge_when_no_stop_was_asked_for():
    """There is nothing to blame and nothing to drop, so the answer stands.

    `test_a_model_that_answers_with_nothing_is_not_an_error` is the contract; this
    guards it against the fallback reaching further than the case it is for.
    """
    chat, completions = chat_over_fake_completions([""])

    assert chat(PROMPT) == ""
    assert len(completions.calls) == 1, "an empty answer should not have been retried"


def test_the_fallback_fires_once_and_then_lets_the_failure_stand():
    """Dropping the stop sequence is one request, not an escape from failing.

    An endpoint that answers nothing whatever the request looks like must still
    spend the retry budget and raise, or a broken endpoint would look like a
    quirk being handled.
    """
    chat, completions = chat_over_fake_completions(None, completions=AlwaysTruncates([]))

    with pytest.raises(LLMCallFailed):
        chat(PROMPT, stop_strs=["\n"])

    assert len(completions.calls) == 5, "the retry budget should still be spent"
    assert [call["stop"] for call in completions.calls].count(["\n"]) == 1, (
        "the stop sequence should have been dropped after the first attempt, and once"
    )


class AlwaysTruncates(StopTruncatesReasoningCompletions):
    """Reasoning with no content whatever the request asks for.

    What too small a `max_completion_tokens` looks like: dropping the stop
    sequence cannot help, because the budget itself is the wall. Records the
    request as it actually arrived, so a caller can see the stop sequence being
    dropped.
    """

    def create(self, **kwargs):
        self.calls.append(kwargs)
        self.truncations += 1
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(
                content=None, reasoning_content=self.reasoning
            ))],
            usage=SimpleNamespace(
                prompt_tokens=self.prompt_tokens, completion_tokens=self.completion_tokens
            ),
        )


def test_reasoning_with_no_answer_is_told_from_an_endpoint_that_said_nothing(capsys):
    """The two are different problems and the log has to tell them apart.

    Once the stop sequence is out of the picture, reasoning present with no
    content means the token budget went on reasoning and left nothing to answer
    with - not that the endpoint returned an empty answer.
    """
    chat, _ = chat_over_fake_completions(None, completions=AlwaysTruncates([]))
    chat._sends_stop = False   # already learned, so the fallback cannot fire again

    with pytest.raises(LLMCallFailed):
        chat(PROMPT, stop_strs=['\n'])

    reported = capsys.readouterr().err
    assert 'max_completion_tokens' in reported, (
        f'the cause is not named, so the fix is not obvious from the log: {reported[:200]}'
    )
    assert 'Full response' in reported, 'the response is the evidence and is not shown'


def test_an_endpoint_that_answers_nothing_at_all_says_so_instead(capsys):
    chat, _ = chat_over_fake_completions([None])

    with pytest.raises(LLMCallFailed):
        chat(PROMPT, stop_strs=['\n'])

    reported = capsys.readouterr().err
    assert 'no content and no reasoning' in reported
    assert 'max_completion_tokens' not in reported, 'this is not a token budget problem'
