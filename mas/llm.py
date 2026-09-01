import sys

from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)
from openai import OpenAI
from dataclasses import dataclass, fields
from abc import ABC, abstractmethod

from .settings import LLMSettings, default_llm_settings
from datetime import datetime


def _refuses_temperature(error: BaseException) -> bool:
    """Whether an endpoint error is a complaint about the temperature parameter.

    Matched on the message, not a status code: OpenAI-compatible servers do not
    agree on one for an unsupported parameter.
    """
    return "temperature" in str(error).lower()


class EndpointUnusable(RuntimeError):
    """The endpoint cannot serve the model a run is about to ask it for.

    Raised before any experiment starts, so a wrong model name or an unreachable
    server costs a second rather than an allocation.
    """


class LLMCallFailed(RuntimeError):
    """Every retry of an LLM call was exhausted without a usable answer.

    Distinct from a model that answered with an empty string, which is returned
    as the answer it is. Callers need to tell the two apart to decide whether
    retrying could help.
    """


@dataclass
class TokenTracker:
    """Per-instance token accounting, scoped to whatever owns it (typically one GPTChat)."""

    completion_tokens: int = 0
    prompt_tokens: int = 0
    intrinsic_completion_tokens: int = 0
    intrinsic_prompt_tokens: int = 0

    def record(self, prompt_tokens: int, completion_tokens: int, intrinsic: bool = False) -> None:
        self.prompt_tokens += prompt_tokens
        self.completion_tokens += completion_tokens
        if intrinsic:
            self.intrinsic_prompt_tokens += prompt_tokens
            self.intrinsic_completion_tokens += completion_tokens

    def snapshot(self) -> "TokenTracker":
        """The counts as they stand, detached from further recording."""
        return TokenTracker(**{field.name: getattr(self, field.name) for field in fields(self)})

    def since(self, earlier: "TokenTracker") -> "TokenTracker":
        """What has been spent since `earlier` was snapshotted."""
        return TokenTracker(**{
            field.name: getattr(self, field.name) - getattr(earlier, field.name)
            for field in fields(self)
        })


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

# None on any of these means "whatever the settings say".
class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        intrinsic: bool = False # pass intrinsic flag to count tokens used by intrinsic memory
    ) -> str:
        pass

class LLM(ABC):
    
    def __init__(self, model_name: str):
        self.model_name: str = model_name

    @abstractmethod
    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        intrinsic: bool = False
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(
        self,
        model_name: str,
        tracker: Optional["TokenTracker"] = None,
        settings: Optional[LLMSettings] = None,
    ):
        super().__init__(model_name=model_name)
        self.settings: LLMSettings = settings if settings is not None else default_llm_settings()
        self.client = OpenAI(
            base_url=self.settings.api_base,
            api_key=self.settings.api_key
        )
        self.tracker: TokenTracker = tracker if tracker is not None else TokenTracker()
        self._sends_temperature: bool = True

    def _create(
        self,
        messages: List[dict],
        temperature: float,
        max_tokens: int,
        stop_strs: Optional[List[str]],
    ):
        """One request, dropping the temperature if the endpoint refuses it.

        A refusal is remembered, and absorbed here rather than spending one of the
        caller's retries.
        """
        request = dict(
            model=self.model_name,
            messages=messages,
            max_completion_tokens=max_tokens,
            stop=stop_strs,
        )

        if self._sends_temperature:
            try:
                return self.client.chat.completions.create(temperature=temperature, **request)
            except Exception as error:
                if not _refuses_temperature(error):
                    raise
                self._sends_temperature = False
                print(
                    f"{self.model_name} refused temperature={temperature} ({error}); "
                    f"sending subsequent calls without it",
                    file=sys.stderr,
                )

        return self.client.chat.completions.create(**request)

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        intrinsic: bool = False,
    ) -> str:
        import time

        if max_tokens is None:
            max_tokens = self.settings.max_tokens
        if temperature is None:
            temperature = self.settings.temperature

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5
        wait_time = 1
        last_error: Optional[BaseException] = None

        for attempt in range(max_retries):
            try:
                response = self._create(
                    messages=messages,
                    temperature=temperature,
                    max_tokens=max_tokens,
                    stop_strs=stop_strs,
                )

                answer = response.choices[0].message.content
                self.tracker.record(
                    prompt_tokens=response.usage.prompt_tokens,
                    completion_tokens=response.usage.completion_tokens,
                    intrinsic=intrinsic,
                )

                if answer is None:
                    print("Error: LLM returned None", file=sys.stderr)
                    continue
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"==== LLM RESPONSE ====\nTIME: {current_time}\n{answer}\n==== END LLM RESPONSE ====\n", file=sys.stderr)
                return answer  

            except Exception as e:
                last_error = e
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    print(f"Rate limited, waiting {wait_time}s before retry {attempt + 2}/{max_retries}", file=sys.stderr)
                    time.sleep(wait_time)
                    wait_time *= 2
                else:
                    print(f"Error during API call: {error_message}", file=sys.stderr)
                    break

        raise LLMCallFailed(
            f'{self.model_name} returned no answer after {max_retries} attempts'
        ) from last_error



def check_endpoint(
    model_name: str,
    settings: Optional[LLMSettings] = None,
    client=None,
) -> None:
    """Fail now if the endpoint cannot serve `model_name`.

    Three questions, in the order a cluster gets them wrong: is anything
    answering, is it serving the model this sweep asked for, and can it generate.
    `client` is for testing without a server.
    """
    chat = GPTChat(model_name=model_name, settings=settings)
    if client is not None:
        chat.client = client

    try:
        served = sorted(model.id for model in chat.client.models.list().data)
    except Exception as error:
        raise EndpointUnusable(
            f'{chat.settings.api_base} answered no model list: {error}'
        ) from error

    if model_name not in served:
        raise EndpointUnusable(
            f"{chat.settings.api_base} does not serve '{model_name}'. "
            f"It serves: {', '.join(served) or 'nothing'}"
        )

    try:
        chat([Message('user', 'ping')], max_tokens=1)
    except Exception as error:
        raise EndpointUnusable(
            f"{chat.settings.api_base} serves '{model_name}' but could not answer: {error}"
        ) from error
