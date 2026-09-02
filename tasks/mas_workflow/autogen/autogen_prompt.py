from dataclasses import dataclass

solver_system_prompt: str = """
You are a smart agent designed to solve problems.
"""

validator_system_prompt: str = """
You are an agent designed to validate the output of the solver agent. 
When you are called, you must evaluate the solver agent's output and determine if it followed the rules set in the reference cases.
ONLY EVALUATE THE FORMAT, NOT THE FACTUAL CORRECTNESS OF THE SOLUTION.
If the solver agent's output format is incorrect, respond with a string formatted f"INVALID: {a brief explanation of why the formatting is incorrect}"
If the solver agent's output format is correct, respond with "VALID". 
"""

ground_truth_system_prompt: str = """
You are an agent designed to assist the solver agent. When you are called, it means the solver agent has repeatedly output the same incorrect content (It means that the solver agent is stuck in a loop of providing the same incorrect answer or approach). 
Your task is to carefully analyze the input and provide the correct answer or guidance to help the solver agent break out of the stuck state and proceed toward the correct solution.

NOTE: ** Your approach must avoid being consistent with the previous output's approach (as the previous output comes from a solver agent that has already fallen into a misconception, making it definitely wrong). **
"""

validator_user_prompt: str = """
                Solver's latest response: \n
                {action} \n 
                Task description: \n
                {task_description} \n
                Format that solver agent's actions must follow: \n
                {few_shots}
                """

solver_revision_prompt: str = """Your response does not follow the expected format. \n 
                    Modify your response according to the Validator's feedback. \n 
                    Your original response: \n
                    {action} \n
                    Validator's feedback: \n 
                    {evaluation} \n 
                    Original instructions: \n
                    """


@dataclass
class AutoGenPrompt:
    solver_system_prompt: str = solver_system_prompt
    ground_truth_system_prompt: str = ground_truth_system_prompt
    validator_system_prompt: str = validator_system_prompt
    validator_user_prompt: str = validator_user_prompt
    solver_revision_prompt: str = solver_revision_prompt

AUTOGEN_PROMPT = AutoGenPrompt()

