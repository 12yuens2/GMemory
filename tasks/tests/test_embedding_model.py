"""The local embedding model is loaded only if something embeds, and on the CPU.

It is the baselines' semantic retrieval: voyager, memorybank, generative and
metagpt hand it to a Chroma store, and g-memory also calls embed_query directly.
The other seven registered modules - empty, chatdev and the five intrinsic
variants - never embed anything.

Both properties are about a GPU node: SentenceTransformer takes cuda:0 when CUDA
is there, and on these nodes every GPU belongs to the vLLM server under test.
"""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from mas import utils
from mas.utils import EmbeddingFunc


@pytest.fixture
def loaded_models(monkeypatch):
    """A stand-in for sentence_transformers, recording every model it loads."""
    loaded: list[tuple] = []

    class FakeSentenceTransformer:
        def __init__(self, model_type, device=None):
            loaded.append((model_type, device))

        def encode(self, text):
            return SimpleNamespace(tolist=lambda: [0.0, 1.0])

    module = ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)

    utils._EMBEDDING_MODEL_CACHE.clear()
    yield loaded
    utils._EMBEDDING_MODEL_CACHE.clear()


def test_constructing_an_embedding_function_loads_no_model(loaded_models):
    """build_mas constructs one for every experiment, before knowing the module."""
    EmbeddingFunc("some-model")

    assert loaded_models == [], "torch and a model load, for a module that never embeds"


def test_the_model_is_loaded_on_first_use(loaded_models):
    embed = EmbeddingFunc("some-model")

    embed.embed_query("a task description")

    assert len(loaded_models) == 1


def test_the_model_is_loaded_once_however_often_it_is_used(loaded_models):
    embed = EmbeddingFunc("some-model")

    embed.embed_query("one")
    embed.embed_documents(["two", "three"])

    assert len(loaded_models) == 1


def test_the_configured_device_is_what_the_model_is_loaded_on(loaded_models):
    """Left to itself it takes cuda:0, which is the server's."""
    EmbeddingFunc("some-model", device="cpu").embed_query("a task description")

    assert loaded_models == [("some-model", "cpu")]


def test_two_devices_are_two_models(loaded_models):
    EmbeddingFunc("some-model", device="cpu").embed_query("one")
    EmbeddingFunc("some-model", device="cuda:1").embed_query("one")

    assert [device for _, device in loaded_models] == ["cpu", "cuda:1"]
