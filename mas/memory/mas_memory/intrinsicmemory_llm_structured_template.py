from dataclasses import dataclass
import logging

from .prompt import INTRINSICMEMORY_LLM_TEMPLATE
from .intrinsicmemory import IntrinsicMASMemory
from ..common import MASMessage # a MASMessage, which is a specific type of message used in MAS
from mas.llm import Message # a "normal" message, not a MASMessage?

logger = logging.getLogger(__name__)

@dataclass
class IntrinsicMASMemoryLLMTemplate(IntrinsicMASMemory):
    """
    IntrinsicMASMemoryLLMTemplate keeps agent-specific memory using automatically generated structured memory templates with LLMs.
    
    Task context is stored persistently in mas_message, which includes the task 
    description (mas_message.task_description) and trajectory (mas_message.task_trajectory) (i.e., the sequence of actions taken to complete the task).)

    A separate memory is used to store the agent's memory, which is updated with the latest information from the agent's messages.
    
    """
    system_prompt = INTRINSICMEMORY_LLM_TEMPLATE.system_prompt

    def __post_init__(self):
        super().__post_init__()
        self.memory_template: str = ""
        self.memory_template_flag: bool = False


    def summarize(self, *, solver_message: str = "", template_instructions: str = "") -> str:
        """Update the agent's memory, generating the template on the first call.

        A caller's `template_instructions` is ignored in favour of the generated one.
        """
        mas_message: MASMessage = self.current_task_context
        if not self.memory_template_flag:
            # Generate initial memory template if this is the first time running summarize
            logger.info('generating the initial memory template')

            template_creation_message: str = INTRINSICMEMORY_LLM_TEMPLATE.template_creation_prompt.format(
                task_description = mas_message.task_description
            )

            logger.debug('GENERATE MEMORY TEMPLATE\n%s', template_creation_message)
            msg =[Message("system", self.system_prompt), Message("user", template_creation_message)]

            self.memory_template = self.llm_model(msg, intrinsic=True)
            self.memory_template_flag = True

        return super().summarize(
            solver_message=solver_message,
            template_instructions=INTRINSICMEMORY_LLM_TEMPLATE.memory_template_section.format(
                template_instructions=self.memory_template
            ),
        )

