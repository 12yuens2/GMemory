"""Fakes shared by the contract and smoke tests.

Everything here stands in for something that needs a network, a GPU or a
simulator binary, so the suite can exercise real workflow, recorder and memory
code paths offline. Nothing here is a mock with recorded expectations - each
fake is a working, minimal implementation of the interface the production code
is entitled to assume.
"""

import hashlib

from mas.llm import Message, TokenTracker
from mas.reasoning import ReasoningConfig, ReasoningIO


class RunawayLoop(AssertionError):
    """A fake was called far more often than any bounded loop would need."""


class FakeLLM:
    """A GPTChat stand-in that returns scripted replies and counts tokens.

    `replies` is cycled, so a single-element list is an always-the-same LLM and
    an empty string is the failure the retry loops are built around.

    `max_calls` is a runaway guard, not a behaviour: it turns a test that would
    hang on an unbounded retry loop into one that fails in milliseconds with a
    message naming the cause.
    """

    def __init__(self, replies=None, model_name="fake-model", tracker=None, max_calls=200):
        self.replies = list(replies) if replies else ["look at desk 1"]
        self.model_name = model_name
        self.tracker = tracker if tracker is not None else TokenTracker()
        self.max_calls = max_calls
        self.calls: list[list[Message]] = []

    def __call__(self, messages, temperature=None, max_tokens=None,
                 stop_strs=None, num_comps=None, intrinsic=False):
        if len(self.calls) >= self.max_calls:
            raise RunawayLoop(
                f"FakeLLM was called {self.max_calls} times - the caller is not bounding its retries"
            )
        self.calls.append(list(messages))
        self.tracker.record(prompt_tokens=10, completion_tokens=5, intrinsic=intrinsic)
        return self.replies[(len(self.calls) - 1) % len(self.replies)]


def fake_reasoning(replies=None) -> ReasoningIO:
    """The real ReasoningIO over a FakeLLM - the workflows type-check for it."""
    return ReasoningIO(llm_model=FakeLLM(replies=replies))


class FakeEnv:
    """A deterministic environment satisfying the BaseEnv interface.

    Reports `done` after `steps_to_done` steps, so a test can choose between an
    episode that solves, one that exhausts the trial budget, and one that never
    terminates.
    """

    def __init__(self, max_trials: int = 3, steps_to_done: int = 1, reward: float = 1.0):
        self.max_trials = max_trials
        self.steps_to_done = steps_to_done
        self.reward = reward
        self.actions: list[str] = []
        self.reset()

    def reset(self) -> None:
        self.actions = []
        self.done = False

    def set_env(self, task_config: dict) -> tuple[str, str]:
        return "fake task main", "fake task description"

    def step(self, action: str) -> tuple[str, float, bool]:
        self.actions.append(action)
        self.done = len(self.actions) >= self.steps_to_done
        return f"observation after {action}", (self.reward if self.done else 0.0), self.done

    @classmethod
    def process_action(cls, action: str) -> str:
        return action.strip()

    def feedback(self) -> tuple[float, bool, str]:
        return (
            (self.reward, True, "You successfully finished this task!")
            if self.done
            else (0.0, False, "You failed the task.")
        )


class FakeEmbeddingFunc:
    """A deterministic embedding, so cosine similarity is real arithmetic.

    DyLAN's edge weighting feeds these vectors straight into np.dot and
    np.linalg.norm, so a MagicMock would not do - this returns actual floats
    derived from a hash of the text, which is stable across runs and processes.
    """

    dims = 16

    def __init__(self, model_type: str = "fake-embeddings"):
        self.model_type = model_type

    def _vector(self, text: str) -> list[float]:
        digest = hashlib.sha256(str(text).encode("utf-8")).digest()
        return [digest[i] / 255.0 for i in range(self.dims)]

    def embed_query(self, query: str) -> list[float]:
        return self._vector(query)

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]


class RecordingObserver:
    """Stands in for a recorder where a test only cares that messages arrive."""

    def __init__(self):
        self.messages: list[str] = []

    def log(self, message: str) -> None:
        self.messages.append(str(message))


REASONING_CONFIG = ReasoningConfig(temperature=0, stop_strs=["\n"])
