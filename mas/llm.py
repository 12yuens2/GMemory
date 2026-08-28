import sys

from typing import (
    Protocol, 
    Literal,  
    Optional, 
    List,
)
from openai import OpenAI
from dataclasses import dataclass
from abc import ABC, abstractmethod

from .settings import LLMSettings, default_llm_settings
from datetime import datetime


class LLMCallFailed(RuntimeError):
    """Every retry of an LLM call was exhausted without a usable answer.

    Previously this returned "", indistinguishable from a model that genuinely
    answered with nothing. That ambiguity is what made the agent retry loops spin:
    they treated "" as "try again", and an API error produced "" on every attempt.
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


@dataclass(frozen=True)
class Message:
    role: Literal["system", "user", "assistant"]
    content: str

# None means "whatever the settings say"; these used to be module constants read
# from configs.yaml at import time, which bound the process to one working
# directory before any caller had a say.
class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: Optional[int] = None,
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
        num_comps: Optional[int] = None,
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
        # Credentials are read here, on first construction, rather than when this
        # module is imported - so importing mas needs no environment at all.
        self.settings: LLMSettings = settings if settings is not None else default_llm_settings()
        self.client = OpenAI(
            base_url=self.settings.api_base,
            api_key=self.settings.api_key
        )
        self.tracker: TokenTracker = tracker if tracker is not None else TokenTracker()

    def __call__(
        self,
        messages: List[Message],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        stop_strs: Optional[List[str]] = None,
        num_comps: Optional[int] = None,
        intrinsic: bool = False,
    ) -> str:
        import time

        if max_tokens is None:
            max_tokens = self.settings.max_tokens
        if num_comps is None:
            num_comps = self.settings.num_comps

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5
        wait_time = 1
        last_error: Optional[BaseException] = None

        for attempt in range(max_retries):
            try:
                response = self.client.chat.completions.create(
                    model=self.model_name,  
                    messages=messages,
                    max_completion_tokens=max_tokens,
                    #temperature=temperature,
                    n=num_comps,
                    stop=stop_strs
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
                    # The wait was fixed at 1s for every retry, so five attempts
                    # took five seconds and gave a throttled endpoint no room.
                    wait_time *= 2
                else:
                    print(f"Error during API call: {error_message}", file=sys.stderr)
                    break

        raise LLMCallFailed(
            f'{self.model_name} returned no answer after {max_retries} attempts'
        ) from last_error

