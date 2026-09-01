"""Prompts for MacNet memory."""

from dataclasses import dataclass

task_context = """
## Here is the task trajectory:
{task_trajectory}

## Here are the outputs from your upstream nodes and the feedback provided by the environment:
{upstream_outputs}

Please provide your response based on the task trajectory and the output from your upstream node:
"""

node_info = """
----------------
### name: {name}

### action: {action}

### feedback from the environment: {observation}
----------------
"""

@dataclass
class MacNet:
    task_context: str = task_context
    node_info: str = node_info

MACNET = MacNet()
