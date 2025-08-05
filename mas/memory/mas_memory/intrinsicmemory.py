from dataclasses import dataclass
import os
import sys

from .memory_base import MASMemoryBase
from .prompt import INTRINSICMEMORY
from ..common import MASMessage,AgentMessage # a MASMessage, which is a specific type of message used in MAS
from mas.llm import Message, GPTChat # a "normal" message, not a MASMessage?

@dataclass
class IntrinsicMASMemory(MASMemoryBase):
    """
    IntrinsicMASMemory keeps agent-specific memory. 
    
    Task context is stored persistently in mas_message, which includes the task 
    description (mas_message.task_description) and trajectory (mas_message.task_trajectory) (i.e., the sequence of actions taken to complete the task).)

    A separate memory is used to store the agent's memory, which is updated with the latest information from the agent's messages.
    
    """
    def __post_init__(self):
        super().__post_init__()
        os.makedirs(self.persist_dir, exist_ok=True)
        self.counter: int = 0
        self.agent_intrinsic_memory: str = ""

    def summarize(self, solver_message="") -> str:

        """UPDATE AGENT MEMORY STEP"""

        # Update agent's memory using existing agent memory, latest output from the agent, and the memory update prompt
        mas_message: MASMessage = self.current_task_context
        if self.current_task_context is None:
            raise RuntimeError('The current task memory is empty.')

        # Construct the user prompt for memory update with memory update prompt, task description, and latest agent output
        #MEMORY_UPDATE_PROMPT = INTRINSICMEMORY.memory_update_prompt
        #TASK_DESCRIPTION = f'"TASK DESCRIPTION: {mas_message.task_description}'
        #AGENT_MEMORY = f'CURRENT AGENT MEMORY: {self.agent_intrinsic_memory}'
        #LATEST_AGENT_OUTPUT = f'LATEST AGENT OUTPUT: {mas_message.task_trajectory}'
        #memory_update_user_prompt: list[Message] = [Message('system', MEMORY_UPDATE_PROMPT), Message('user', TASK_DESCRIPTION), Message('user', AGENT_MEMORY), Message('user', LATEST_AGENT_OUTPUT)]

        user_prompt = INTRINSICMEMORY.memory_update_prompt.format(
                solver_system_message=solver_message,
                task_description=mas_message.task_description,
                task_trajectory=mas_message.task_trajectory,
                #current_memory=self.agent_intrinsic_memory,
        )

        messages = [Message("system", INTRINSICMEMORY.memory_system_prompt), Message("user", user_prompt)]


        print(f"----MEMORY----\n{user_prompt}\n----END MEMORY----\n", file=sys.stderr)

        # only summarise after some history has been built up
        if len(mas_message.task_trajectory) > 5:
            self.agent_intrinsic_memory = self.llm_model(messages)

        print(f"==== NEW MEMORY ====\n{self.agent_intrinsic_memory}\n==========", file=sys.stderr)
        injection = "You can only perform one action. Output in a single line your next action"

        return mas_message.task_description + "\n\n" + self.agent_intrinsic_memory + "\n\n" + injection


    def save_task_context(self, label: bool, feedback: str = None) -> MASMessage:
        # Task is finished so wipe current memory
        self.counter = 0
        self.agent_intrinsic_memory = ""

        # reset self.llm_model
        llm_model_name = self.llm_model.model_name
        self.llm_model = GPTChat(model_name=llm_model_name)

    # chat history
    # context
    # agent memory
    # agent memory update
    # agent memory initialisation
