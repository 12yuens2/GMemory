"""Prompt constants, one module per memory module.

Every prompt here belongs to the memory module its file is named for, so a
constant edited for one memory system cannot silently be the one another reads.
The intrinsic variants share `intrinsic.py`: they differ only in the memory
template their system prompt asks for.
"""

from .chatdev import CHATDEV, ChatDev
from .generative import GENERATIVE, Generative
from .gmemory import GMemoryPrompt, GMemoryPrompts
from .intrinsic import (
    INTRINSICMEMORY_ALFWORLD,
    INTRINSICMEMORY_DEFAULT,
    INTRINSICMEMORY_FEVER,
    INTRINSICMEMORY_LLM_TEMPLATE,
    INTRINSICMEMORY_NOTEMPLATE,
    INTRINSICMEMORY_PDDL,
    MEMORY_TEMPLATE_SECTION,
    MEMORY_UPDATE_PROMPT,
    IntrinsicMemoryALFWORLD,
    IntrinsicMemoryDefault,
    IntrinsicMemoryFEVER,
    IntrinsicMemoryLLMTemplate,
    IntrinsicMemoryNoTemplate,
    IntrinsicMemoryPDDL,
)
from .macnet import MACNET, MacNet
from .memorybank import MEMORYBANK, MemoryBank
from .voyager import VOYAGER, Voyager

__all__ = [
    'CHATDEV',
    'ChatDev',
    'GENERATIVE',
    'Generative',
    'GMemoryPrompt',
    'GMemoryPrompts',
    'INTRINSICMEMORY_ALFWORLD',
    'INTRINSICMEMORY_DEFAULT',
    'INTRINSICMEMORY_FEVER',
    'INTRINSICMEMORY_LLM_TEMPLATE',
    'INTRINSICMEMORY_NOTEMPLATE',
    'INTRINSICMEMORY_PDDL',
    'IntrinsicMemoryALFWORLD',
    'IntrinsicMemoryDefault',
    'IntrinsicMemoryFEVER',
    'IntrinsicMemoryLLMTemplate',
    'IntrinsicMemoryNoTemplate',
    'IntrinsicMemoryPDDL',
    'MACNET',
    'MacNet',
    'MEMORYBANK',
    'MemoryBank',
    'MEMORY_TEMPLATE_SECTION',
    'MEMORY_UPDATE_PROMPT',
    'VOYAGER',
    'Voyager',
]
