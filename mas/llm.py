import os
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

from .utils import load_config
from datetime import datetime


# model configs
CONFIG: dict = load_config("configs/configs.yaml")
LLM_CONFIG: dict = CONFIG.get("llm_config", {})
MAX_TOKEN = LLM_CONFIG.get("max_token", 512)  
TEMPERATURE = LLM_CONFIG.get("temperature", 0.1)
NUM_COMPS = LLM_CONFIG.get("num_comps", 1)

URL = os.environ["OPENAI_API_BASE"]
KEY = os.environ["OPENAI_API_KEY"]
#print('# api url: ', URL)
#print('# api key: ', KEY)


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

class LLMCallable(Protocol):

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
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
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
        intrinsic: bool = False
    ) -> str:
        pass

class GPTChat(LLM):

    def __init__(self, model_name: str, tracker: Optional["TokenTracker"] = None):
        super().__init__(model_name=model_name)
        self.client = OpenAI(
            base_url=URL,
            api_key=KEY
        )
        self.tracker: TokenTracker = tracker if tracker is not None else TokenTracker()

    def __call__(
        self,
        messages: List[Message],
        temperature: float = TEMPERATURE,
        max_tokens: int = MAX_TOKEN,
        stop_strs: Optional[List[str]] = None,
        num_comps: int = NUM_COMPS,
        intrinsic: bool = False,
    ) -> str:
        import time

        messages = [{"role": msg.role, "content": msg.content} for msg in messages]

        max_retries = 5  
        wait_time = 1 

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
                    print("Error: LLM returned None")
                    continue
                current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                print(f"==== LLM RESPONSE ====\nTIME: {current_time}\n{answer}\n==== END LLM RESPONSE ====\n", file=sys.stderr)
                return answer  

            except Exception as e:
                error_message = str(e)
                if "rate limit" in error_message.lower() or "429" in error_message:
                    time.sleep(wait_time)
                else:
                    print(f"Error during API call: {error_message}")
                    break 

        return ""

