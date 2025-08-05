from dataclasses import dataclass
#----------------------------------------------intrinsicmemory memory----------------------------------------------
DEFAULT_MEMORY_UPDATE_PROMPT = """
Use the entire history of the current task trajectory to populate and update the current memory with factual information to solve the task based on the task descrpition.

The task trajectory shows the full history of previous actions, each action is followed by the environment observation in sequence:
[action]
[observation]
[action]
[observation]
...

## Task Description
{task}

## Current Task Trajectory
{task_trajectory}

{current_memory}

## Instructions
Update the current memory by summarising the current task trajectory. Output only the memory format as shown below and nothing else:

## Current Memory
Previous Actions:
Current Status/Environment:
Next steps:
"""

DEFAULT_MEMORY_UPDATE_PROMPT="""
You will receive a task description and a sequence of action-observation pairs representing a task trajectory. Your job is to summarize what has actually happened based solely on this data, avoiding assumptions.

Each pair is in the format:

action  
[environment response]

The actions are attempts by the agent. The observations are what the environment reports in response.

## Rules:
- Carefully read both the action **and** the corresponding environment response.
- Do not assume an action succeeded unless the observation confirms it. For example:
    - If the response is vague, says nothing occurred, indicates an error, or provides no change in the environment (e.g. "Nothing happens", "No result", "Invalid action", "That doesn't work", etc.), then the action **did not succeed**.
    - If nothing occurred, do not assume any facts from the failed action
- Track failed attempts — they may be useful later to avoid redundancy (e.g., repeated failure may mean a missing precondition).
- If an interaction repeatedly fails (e.g. taking an item, search with no results), **do not recommend retrying** unless there's a logical reason it might now succeed 
- Base your summary only on actual evidence. Do not invent or infer missing steps.
- Always reflect the current state both the task description and what has (and hasn’t) worked so far.
- Follow the exact output format below. Use no more than 10 lines/sentences.

---

## Task Description
{solver_system_message}
{task_description}

## Task Trajectory  
{task_trajectory}

---

## Output Format

## Current Memory  
Previous Actions:  
[Summarised list of previous actions and whether they succeeded or failed. Include what was attempted and the environment response.]

Current Status/Environment:  
[Summarize what is currently known, based only on environment responses — location, visible objects, any relevant state.]

---

Now, summarize the current memory based on the task trajectory and description.
"""

#keeping the most relevant information for the current task to inform me what's been done and what to do next.
#- If the "Current Task Trajectory" is empty that means nothing has happened yet. There have been no previous actions. Just return the current memory format empty.
#- The "Current Task Trajectory" contains the absolute truth of the entire history of what has happened, if the current memory is incorrect compared to the current task trajectory, update and correct it. For example if the current memory contains previous actions but the current task trajectory contradicts those actions.
#- DO NOT hallucinate previous actions if they have not happened according to the current task trajectory
#- If current memory exists, it may have already summarised earlier parts of the task trajectory, so focus on updating the memory with the newest actions and observations.
#- 'think' or 'thought' actions do not count as actions that have been done yet, the response from the environment determines what's been done, but you can summarize what's been thought about.
#- Keep the summary concise in no more than 5 sentences.



DEFAULT_MEMORY_SYSTEM_PROMPT = """
You are an intelligent summarization agent. Your job is to summarize a set of task trajectories in which I have performed a sequence of actions, and the environment has provided corresponding observations. You must convert this information into a memory that is useful for efficient task completion.

- Use only the information explicitly present in the provided task trajectory. Do not invent or infer any actions, events, or observations that are not stated.
- If no actions or observations have occurred yet, state this clearly.
- Never invent or assume outcomes. Stick only to observed events and logical inferences grounded in the sequence.
- Maintain contextual consistency so that I can continue working on the task without confusion.
- Ensure the summarized memory makes clear what has already been completed, what the environment currently contains (based only on observation)
- Use clear and precise language. Avoid unnecessary details or storytelling.
- Output memory in this format only:

## Current Memory
Previous Actions: [Bullet or numbered list of grounded actions. If none, write: “None.”]
Current Status/Environment: [Based only on last observed environment. If no observations, use the task description.]
""" 


@dataclass
class IntrinsicMemory:
    memory_update_prompt: str = DEFAULT_MEMORY_UPDATE_PROMPT
    memory_system_prompt: str = DEFAULT_MEMORY_SYSTEM_PROMPT

INTRINSICMEMORY: IntrinsicMemory = IntrinsicMemory()


# ---------------------------------------------- ChatDev memory ----------------------------------------------
summary_system_prompt: str = """
You are an agent skilled in summarization. Your task is to generate **phase-based summaries** from given execution records of an agent's task. These summaries help the agent efficiently utilize existing information, avoid redundant computations, and ensure task continuity.

## **Requirements for Your Summary:**
1. **Phase-based summarization**: Organize execution records into logical phases and extract key steps.
2. **Task relevance**: Ensure the summary helps the agent understand what has been completed and what needs to be done next.
3. **Clarity and conciseness**: Use clear and precise language to summarize the information while avoiding unnecessary details.

## **Additional Guidelines:**
- Maintain **contextual consistency** so that the agent can seamlessly continue the task.
- If there are incorrect intermediate states or irrelevant information, filter or correct them to make the summary more accurate.
"""


summary_user_prompt: str = """You will be given a partial execution record of an agent's task. Your job is to generate a **phase-based summary** that the agent can understand and use to continue the task.

## **Your Summary Should Follow These Guidelines:**
1. **Phase-based summarization**: Break the record into logical steps, ensuring that each phase's key tasks are captured.
2. **Efficient information transfer**:
   - Document key task objectives, executed actions, and the current state.
   - Identify unfinished parts to help the agent determine the next steps.
3. **Prevent information loss**:
   - Include critical decision points, state changes, and key computation processes.
   - If there are uncertainties, retain relevant details for future judgment.

---

## **Example:**
Please strictly follow the output format of the example!
### **Input (Partial Execution Record)**
1. Task Objective: Classify news articles.
2. Preprocessing: Remove stopwords, tokenize, and normalize text.
3. Feature Extraction: Compute TF-IDF vectors.
4. Model Training: Tried SVM and RandomForest.
5. Evaluation: SVM's F1-score is 0.82, while RandomForest's is 0.78.

### **Output**
Done: Completed text cleaning (stopword removal, tokenization, normalization). Computed TF-IDF feature vectors. Trained SVM and RandomForest classifiers. Evaluated models—SVM achieved an F1-score of 0.82, outperforming RandomForest (0.78).
Next Steps: Perform hyperparameter tuning to improve SVM’s classification performance. Consider exploring deep learning models (e.g., Transformers) for further enhancement. Visualize misclassified samples to analyze model weaknesses.
(Example End)

Now it's your turn, here is the task and its partial execution:
## Task: 
{task}

## Task Trajectory:
{task_trajectory}

Output: 
"""

@dataclass
class ChatDev:
    summary_system_instruction: str
    summary_user_instruction: str

CHATDEV: ChatDev = ChatDev(
    summary_system_instruction=summary_system_prompt,
    summary_user_instruction=summary_user_prompt
)


# ---------------------------------------------- MacNet memory ----------------------------------------------
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

# ---------------------------------------------- Voyager memory ----------------------------------------------
task_sumary_system_instruction = """
You are a helpful assistant that writes a description of the task resolution trajectory.
1) Try to summarize the trajectory in no more than 6 sentences.
2) Your response should be a single line of text."""

task_summary_user_instruction = """
For example:
Trajectory:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some egg and put it in diningtable.
> think: To solve the task, I need to find and take an egg, then heat it with microwave, then put it in diningtable.
OK.
> think: First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
OK.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
> go to countertop 1
On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
> go to countertop 2
On the countertop 2, you see nothing.
> go to countertop 3
On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
> think: Now I find an egg (2). Next, I need to take it.
OK.
> take egg 2 from countertop 3
You pick up the egg 2 from the countertop 3.
> think: Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
OK.
> go to microwave 1
The microwave 1 is closed.
> heat egg 2 with microwave 1
You heat the egg 2 using the microwave 1.
> think: Now I heat an egg (2). Next, I need to put it in/on diningtable 1.
OK.
> go to diningtable 1
On the diningtable 1, you see a apple 2, a bread 3, a egg 1, a kettle 1, a knife 1, a mug 1, a papertowelroll 1, a peppershaker 2, a potato 1, a soapbottle 1, and a spatula 1.
> put egg 2 in/on diningtable 1
You put the egg 2 in/on the diningtable 1.

Then you would write: The trajectory is about finding an egg, heating it with a microwave, and placing it on the dining table after checking various locations like the fridge and countertops.

Trajectory:
{task_trajectory}
"""

@dataclass
class Voyager:
    task_summary_system_instruction: str
    task_summary_user_instruction: str

VOYAGER = Voyager(
    task_summary_system_instruction=task_sumary_system_instruction,
    task_summary_user_instruction=task_summary_user_instruction
)
# ---------------------------------------------- MemoryBank memory ----------------------------------------------
task_sumary_system_instruction = """
You are a helpful assistant that writes a description of the task resolution trajectory.
1) Try to summarize the trajectory in no more than 6 sentences.
2) Your response should be a single line of text."""

task_summary_user_instruction = """
For example:
Trajectory:
You are in the middle of a room. Looking quickly around you, you see a cabinet 10, a cabinet 9, a cabinet 8, a cabinet 7, a cabinet 6, a cabinet 5, a cabinet 4, a cabinet 3, a cabinet 2, a cabinet 1, a coffeemachine 1, a countertop 3, a countertop 2, a countertop 1, a diningtable 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a fridge 1, a garbagecan 1, a microwave 1, a sinkbasin 1, a stoveburner 4, a stoveburner 3, a stoveburner 2, a stoveburner 1, and a toaster 1.
Your task is to: heat some egg and put it in diningtable.
> think: To solve the task, I need to find and take an egg, then heat it with microwave, then put it in diningtable.
OK.
> think: First I need to find an egg. An egg is more likely to appear in fridge (1), countertop (1-3), diningtable (1), stoveburner (1-4), toaster (1), garbagecan (1), cabinet (1-10). I can check one by one, starting with fridge 1.
OK.
> open fridge 1
You open the fridge 1. The fridge 1 is open. In it, you see a lettuce 2, a mug 2, and a potato 3.
> go to countertop 1
On the countertop 1, you see a bread 1, a fork 1, and a saltshaker 1.
> go to countertop 2
On the countertop 2, you see nothing.
> go to countertop 3
On the countertop 3, you see a bowl 1, a butterknife 1, a egg 2, a kettle 2, a plate 1, a sink 1, and a spatula 2.
> think: Now I find an egg (2). Next, I need to take it.
OK.
> take egg 2 from countertop 3
You pick up the egg 2 from the countertop 3.
> think: Now I take an egg (2). Next, I need go to a microwave (1) and heat it.
OK.
> go to microwave 1
The microwave 1 is closed.
> heat egg 2 with microwave 1
You heat the egg 2 using the microwave 1.
> think: Now I heat an egg (2). Next, I need to put it in/on diningtable 1.
OK.
> go to diningtable 1
On the diningtable 1, you see a apple 2, a bread 3, a egg 1, a kettle 1, a knife 1, a mug 1, a papertowelroll 1, a peppershaker 2, a potato 1, a soapbottle 1, and a spatula 1.
> put egg 2 in/on diningtable 1
You put the egg 2 in/on the diningtable 1.

Then you would write: The trajectory is about finding an egg, heating it with a microwave, and placing it on the dining table after checking various locations like the fridge and countertops.

Trajectory:
{task_trajectory}
"""

@dataclass
class MemoryBank:
    task_summary_system_instruction: str
    task_summary_user_instruction: str

MEMORYBANK: MemoryBank = MemoryBank(
    task_summary_system_instruction=task_sumary_system_instruction,
    task_summary_user_instruction=task_summary_user_instruction
)

# ---------------------------------------------- Generative memory ----------------------------------------------
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


# ---------------------------------------------- G-Memory memory ----------------------------------------------
# Retrieve tasks based on task relevance
generative_task_system_prompt = """You are an agent designed to score the relevance between two pieces of text."""
generative_task_user_prompt = '''You will be given a successful case where you successfully complete the task. Then you will be given an ongoing task. Do not summarize these two cases, but rather evaluate how relevant and helpful the successful case is for the ongoing task, on a scale of 1-10.
Success Case:
{trajectory}
Ongoing task:
{query_scenario}
Score: '''

# successful trajectory extraction
extract_true_traj_system_prompt = """You are an agent skilled at extracting key points.
Given a task and a successful execution trajectory, your job is to identify the critical steps needed to complete the task while filtering out less important steps."""
extract_true_traj_user_prompt = """
Note: 
- Strictly follow the original trajectory; absolutely no steps that are not in the trajectory should be added.
- Even in a successful trajectory, there may be some incorrect steps. Pay attention to actions that correspond to "Nothing happens" observations, as these actions are likely incorrect. Filter out these actions for me.
- You need to ensure that each step is at the finest granularity.
- You should strictly follow the output format in the example.

## Example 1:
### Task
You are in the middle of a room. Looking quickly around you, you see a bed 1, a desk 2, a desk 1, a drawer 6, a drawer 5, a drawer 4, a drawer 3, a drawer 2, a drawer 1, a garbagecan 1, a laundryhamper 1, a safe 1, a shelf 6, a shelf 5, a shelf 4, a shelf 3, a shelf 2, and a shelf 1.
Your task is to: look at bowl under the desklamp.

### Trajectory
... The agent manages to find a bowl 1 on desk 2
Act 14: think: Now I find a bowl (1) on desk 2. Next, I need to take it.
Obs 14: OK.
Act 15: take bowl 1 from desk 2
Obs 15: You pick up the bowl 1 from the desk 2.
Act 16: think: Now I take a bowl (1). Next, I need to find a desklamp. A desklamp is more likely to appear in desk (1-2), shelf (1-6), or drawer (1-6). I can check one by one, starting with desk 1 since I already found a desklamp there.
Obs 16: OK.
Act 17: go to desk 1
Obs 17: On the desk 1, you see a desklamp 1, a keychain 3, a mug 2, a pen 2, a pen 1, and a pencil 2.
Act 18: think: Now I find a desklamp (1) on desk 1. Next, I need to use it.
Obs 18: OK.
Act 19: use desklamp 1
Obs 19: You turn on the desklamp 1.
reward: True

### Output
Construct the overall approach: Find and take a bowl, then find and use a desklamp.
Search for the Bowl: Check all locations systematically in order until the bowl is found (e.g., drawers, desks, shelves, garbage can, laundry hamper, and any other possible places).
Find the Bowl: Locate the bowl on desk 2.
Take the Bowl: Pick up the bowl from desk 2.
Search for the Desklamp: Recall that a desklamp was found earlier on desk 1.
Go to Desk 1: Move to desk 1 where the desklamp is located.
Use the Desklamp: Turn on the desklamp.

Now it's your turn! 
## Here is the task:
### Task
{task}

### Trajectory
{trajectory}

### Output
"""

# Insights
finetune_insights_suffix = dict(full = """Focus on REMOVE or EDIT or AGREE rules first, and stop ADD rule unless the new rule is VERY insightful and different from EXISTING RULES.
""", not_full = """""")

format_rules_operation_template = """<OPERATION> <RULE NUMBER>: <RULE> (e.g. ADD: xxx, EDIT/REMOVE/AGREE 1: xxx)

The available operations are: **AGREE (if the existing rule is strongly relevant for the task), REMOVE (if one existing rule is contradictory or similar/duplicated to other existing rules), EDIT (if any existing rule is not general enough or can be enhanced, rewrite and improve it), ADD (add new rules that are very different from existing rules and relevant for other tasks). Each needs to CLOSELY follow their corresponding formatting below (any existing rule not edited, not agreed, nor removed is considered copied)**:

AGREE <EXISTING RULE NUMBER>: <EXISTING RULE>
REMOVE <EXISTING RULE NUMBER>: <EXISTING RULE>
EDIT <EXISTING RULE NUMBER>: <NEW MODIFIED RULE>
ADD: <NEW RULE>

Do not mention the trials in the rules because all the rules should be GENERALLY APPLICABLE. Each rule should be concise and easy to follow. Any operation can be used MULTIPLE times. Do at most 4 operations and each existing rule can only get a maximum of 1 operation. """

#
critique_compare_rules_system_prompt = """
You are an advanced reasoning agent capable of deriving rules based on examples. You will be given two similar tasks: 
the first one is correct, and the second one is incorrect, The reason for failure has already been provided for the failed trajectory.

Requirements:
- Convert the reasons for failure into insights for future agents to reference, in order to avoid making the same mistakes.
- The insights you summarize must follow the "XXX, because XXX" format. They should not mention specific items but should instead extract general success principles applicable to similar tasks. These insights must be enlightening and provide guidance for future problems.  

"""

critique_compare_rules_user_prompt = """
## Trial Task 1 (success):
{task1}
{task1_trajectory}

## Trial Task 2 (fail):
### Failed reason
{fail_reason}

### Trajectory
{task2}
{task2_trajectory}

## Here are the EXISTING RULES:
{existing_rules}

By examining and contrasting to the successful trial, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules is GENERAL and HIGH LEVEL critiques of the failed trial or proposed way of Thought so they can be used to avoid similar failures when encountered with different questions in the future. Have an emphasis on critiquing how to perform better Thought and Action. Follow the below format:
""" + format_rules_operation_template

# all success instruction
critique_success_rules_system_prompt = """You are an advanced reasoning agent that can add, edit or remove rules from your existing rule set, based on forming new critiques of past task trajectories. 
You will be given successful tasks trials in which you were placed in a household environment and tasks to complete."""

critique_success_rules_user_prompt = """
## Requirements:  
- Avoid vague statements; ensure each insight has a clear causal relationship.  
- Focus only on strategies that apply to a broad range of scenarios rather than case-specific advice.  
- Keep the language concise and to the point, ensuring clarity and practical value.  
- The insights you summarize must follow the "XXX, because XXX" format. They should not mention specific items but should instead extract general success principles applicable to similar tasks. These insights must be enlightening and provide guidance for future problems.  

## Examples:  
- Eliminate unnecessary thinking, because focusing on core objectives improves execution efficiency.
  
## Here are the trials:
{success_history}

## Here are the EXISTING RULES:
{existing_rules}

By examining the successful trials, and the list of existing rules, you can perform the following operations: add, edit, remove, or agree so that the new list of rules are general and high level insights of the successful trials or proposed way of Thought so they can be used as helpful tips to different tasks in the future. Have an emphasis on tips that help the agent perform better Thought and Action. Follow the below format:
""" + format_rules_operation_template

# detect mistakes in trajectory
detect_mistakes_system_prompt = """You are an analytical agent. You will be given a task and a failed trajectory.
The reason for the failure is that the final task state does not match the required task state.

You need to carefully analyze the final state of the failed trajectory and compare it with the required task state to identify inconsistencies. Please check:
    - Whether the state of the target object matches the required task state.
    - Whether the target object has been placed in the required location.

Rules:
    - Any object with the same name (even with different numbers) satisfies the task requirements. For example, if the task requires "an apple" and the agent finds "apple 2," it is still considered correct.
    - Your analysis must ignore numbers and ensure matching is based only on names and states.

Based on the above rules, summarize the most likely reason for the error in a concise manner."""

detect_mistakes_user_prompt = """
Identify the inconsistency between the final state of the target in the incorrect trajectory and the required task goal. 
This is a failed trajectory.:
## Task
{task}

## Trajectory
{trajectory}

Your output:
"""

# merge rules
merge_rules_system_prompt = """You are an agent skilled at summarizing and distilling insights. You are given a list of insights that were previously extracted from similar tasks. These insights may contain redundancy or overlap.

Your job is to **merge and consolidate similar insights**, and output a refined version that is **clear, actionable, and concise**.

NOTE:
- All merged insights **must be based strictly on the given inputs**. You are **not allowed to make up** or infer any new information.
- The output should be easy to read and follow.

📝 Output Format:
- Start your response directly with the numbered list, no preamble or explanations.
- Each insight should be a short sentence.
- Use the following format exactly:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

merge_rules_user_prompt = """
## Here are the current insights that need to be merged:
{current_rules}

## Please consolidate and rewrite them into **no more than {limited_number} refined insights**.

As the summarizing agent, remove redundancies, combine similar ideas, and ensure clarity.

Your output:
"""

# annalyze patterns
analyze_mas_pattern_system_prompt = """You are an expert at identifying improvements in multi-agent system (MAS) outputs.
Given the initial outputs from several agents and the final output produced by the MAS, your task is to determine whether the MAS output shows any **improvement** over the initial agent outputs.

If there is an improvement, respond with: True
If there is no improvement, respond with: False

Important: Do not include any explanation, formatting, or extra characters — only output True or False.
"""

analyze_mas_pattern_user_prompt = """
### Initial outputs from agents:
{agents_init_outputs}

### Final output from the MAS:
{mas_output}

Does the MAS output show an improvement over the initial agent outputs?
Respond with only True or False:

Your answer:
"""

# project insights according to agent's role
project_insights_system_prompt: str = """
You are a thoughtful and context-aware agent. You will be given a specific agent **role** and a set of **general insights** that apply to all roles. 
Your task is to **adapt these general insights** into **personalized insights tailored to the given role**, helping the agent perform more effectively.
Make sure your output aligns with the role's background, responsibilities, and point of view.

NOTE - Your output should follow the below format:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

project_insights_user_prompt: str = """
### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""

# project insights according to agent's role and trajectory
project_insights_with_traj_system_prompt: str = """
You are a thoughtful and context-aware agent. You will be provided with a successfully executed **trajectory**, a specific agent **role**, and a set of **general insights** applicable across all roles.
Your task is to **adapt these general insights** into **personalized insights** that are specifically tailored to the given role and its trajectory. These personalized insights should help the agent improve future performance by aligning with their unique background, responsibilities, and perspective.
Make sure your output reflects an understanding of the role's context and promotes actionable, role-relevant advice.

NOTE - Your output must strictly follow the format below:
1. Insight 1
2. Insight 2
3. Insight 3
...
"""

project_insights_with_traj_user_prompt: str = """
### Trajectory
{trajectory}

### Agent's Role:
{role}

### General Insights:
{insights}

### Your Output (Personalized Insights for This Role):
"""



@dataclass
class GMemoryPrompt:
    generative_task_system_prompt = generative_task_system_prompt
    generative_task_user_prompt = generative_task_user_prompt
    extract_true_traj_system_prompt = extract_true_traj_system_prompt
    extract_true_traj_user_prompt = extract_true_traj_user_prompt
    finetune_insights_suffix = finetune_insights_suffix
    critique_compare_rules_system_prompt = critique_compare_rules_system_prompt
    critique_compare_rules_user_prompt = critique_compare_rules_user_prompt
    critique_success_rules_system_prompt = critique_success_rules_system_prompt
    critique_success_rules_user_prompt = critique_success_rules_user_prompt
    detect_mistakes_system_prompt = detect_mistakes_system_prompt
    detect_mistakes_user_prompt = detect_mistakes_user_prompt
    merge_rules_system_prompt = merge_rules_system_prompt
    merge_rules_user_prompt = merge_rules_user_prompt
    analyze_mas_pattern_system_prompt=analyze_mas_pattern_system_prompt
    analyze_mas_pattern_user_prompt=analyze_mas_pattern_user_prompt
    project_insights_system_prompt=project_insights_system_prompt
    project_insights_user_prompt=project_insights_user_prompt
    project_insights_with_traj_system_prompt=project_insights_with_traj_system_prompt
    project_insights_with_traj_user_prompt=project_insights_with_traj_user_prompt


GMemoryPrompts = GMemoryPrompt()
