"""A sweep refuses to start against an endpoint that cannot serve its model.

The failure this prevents costs an allocation rather than a test: two of the
Slurm scripts, as written, asked a server for a model it was not serving, and
nothing said so until every experiment had failed separately, hours in.
"""

from types import SimpleNamespace

import pytest

from mas.llm import EndpointUnusable, check_endpoint
from mas.settings import LLMSettings

SETTINGS = LLMSettings(api_base="http://localhost:9999/v1", api_key="none")


class FakeModels:
    def __init__(self, served: list[str]):
        self.served = served

    def list(self):
        return SimpleNamespace(data=[SimpleNamespace(id=name) for name in self.served])


def fake_client(served: list[str], completions=None):
    return SimpleNamespace(
        models=FakeModels(served),
        chat=SimpleNamespace(completions=completions),
    )


class OneTokenCompletions:
    def __init__(self):
        self.calls: list[dict] = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content="ok"))],
            usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
        )


def test_a_model_the_endpoint_does_not_serve_is_refused_by_name():
    client = fake_client(["Qwen/Qwen3.6-35B-A3B"])

    with pytest.raises(EndpointUnusable, match="openai/gpt-oss-120b"):
        check_endpoint("openai/gpt-oss-120b", settings=SETTINGS, client=client)


def test_the_refusal_says_what_the_endpoint_does_serve():
    client = fake_client(["Qwen/Qwen3.6-35B-A3B"])

    with pytest.raises(EndpointUnusable, match="Qwen/Qwen3.6-35B-A3B"):
        check_endpoint("openai/gpt-oss-120b", settings=SETTINGS, client=client)


def test_an_endpoint_that_does_not_answer_at_all_is_refused():
    class Unreachable:
        def list(self):
            raise ConnectionError("connection refused")

    client = fake_client([])
    client.models = Unreachable()

    with pytest.raises(EndpointUnusable, match="http://localhost:9999/v1"):
        check_endpoint("openai/gpt-oss-120b", settings=SETTINGS, client=client)


def test_a_served_model_is_asked_for_one_token_before_the_sweep_starts():
    """Listing a model is not the same as being able to generate with it."""
    completions = OneTokenCompletions()
    client = fake_client(["openai/gpt-oss-120b"], completions=completions)

    check_endpoint("openai/gpt-oss-120b", settings=SETTINGS, client=client)

    assert len(completions.calls) == 1
    assert completions.calls[0]["max_completion_tokens"] == 1
