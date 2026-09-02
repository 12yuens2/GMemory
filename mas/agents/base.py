from typing import Protocol, runtime_checkable

from mas.reasoning import ReasoningBase, ReasoningConfig
from mas.llm import Message


class Agent:
    def __init__(
        self, 
        name: str, 
        role: str,
        system_instruction: str,
        reasoning_module: ReasoningBase,
        memory_module = None
    ):
        if reasoning_module is None:
            raise ValueError("The reasoning module should not be none.")
        
        # basic info
        self.name: str = name 
        self.profile: str = role 
        self.system_instruction: str = system_instruction
        
        # reasoning module
        self.reasoning: ReasoningBase = reasoning_module
        self.memory = memory_module

        self.total_system_instruction: str = self.system_instruction
    
    def add_task_instruction(self, task_instruction: str) -> str:
        self.total_system_instruction = self.system_instruction + '\n' + task_instruction
        return self.total_system_instruction

    def response(self, user_prompt: str, reason_config: ReasoningConfig) -> str:
        messages: list[Message] = [Message('system', self.total_system_instruction), Message('user', user_prompt)]
        return self.reasoning(messages, reason_config)



@runtime_checkable
class Env(Protocol):
    """What a workflow needs of an environment, whoever supplies it.

    Structural, so the test fakes and the tasks/envs implementations satisfy it
    without a common base class. `tasks.envs.BaseEnv` is the implementation side.
    """

    max_trials: int

    def set_env(self, configs: dict) -> tuple[str, str]: ...

    def reset(self) -> None: ...

    def step(self, action: str) -> tuple[str, float, bool]: ...

    def process_action(self, action: str) -> str: ...

    def is_thought(self, action: str) -> bool: ...

    def feedback(self) -> tuple[float, bool, str]: ...