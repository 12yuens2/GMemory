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

from tasks.tests.fakes import (
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


def test_a_none_answer_without_reasoning_content_is_not_treated_as_truncation():
    """The fallback must not mask a genuinely empty answer as a stop-sequence quirk.

    Spending the retry budget is only half of it: what would really be lost is the
    stop sequence, dropped for the life of the client on the strength of an empty
    answer that had nothing to do with it. So every attempt still has to carry it.
    """
    chat, completions = chat_over_fake_completions([None])

    with pytest.raises(LLMCallFailed):
        chat(PROMPT, stop_strs=["\n"])

    assert len(completions.calls) == 5, "a genuinely empty answer should spend the retry budget"
    assert [call["stop"] for call in completions.calls] == [["\n"]] * 5, (
        "the stop sequence was abandoned over an empty answer that never involved it"
    )


class AlwaysTruncates(StopTruncatesReasoningCompletions):
    """Reasoning with no content whatever the request asks for.

    What too small a `max_completion_tokens` looks like: dropping the stop
    sequence cannot help, because the budget itself is the wall.
    """

    def create(self, **kwargs):
        return super().create(**dict(kwargs, stop=['\n']))


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
