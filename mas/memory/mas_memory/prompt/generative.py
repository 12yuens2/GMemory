"""Prompts for Generative memory."""

from dataclasses import dataclass

# generative memory
select_task_system_prompt = """You are an agent designed to score the relevance between two pieces of text."""
select_task_user_prompt = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Your output format should be:
Score: '''

# format task
task_format = """
# Task {id}:
## Task description: 
{task_description}

## Key steps:
{key_steps}

## Detailed trajectory:
{trajectory}
"""

@dataclass
class Generative:
    select_task_system_prompt: str = select_task_system_prompt
    select_task_user_prompt: str = select_task_user_prompt
    task_format: str = task_format

GENERATIVE = Generative()
