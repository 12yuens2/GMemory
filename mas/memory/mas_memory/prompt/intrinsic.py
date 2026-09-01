"""Prompts for the intrinsic memory modules.

One file for the whole family: they share the update procedure and differ only
in the memory template their system prompt asks for, which is the variable the
experiments compare.
"""

from dataclasses import dataclass

#----------------------------------------------intrinsicmemory memory DEFAULT----------------------------------------------

MEMORY_UPDATE_PROMPT = """
Use your latest response to create the new memory with factual information to solve the task based on the task description, current task trajectory, and current memory. OUTPUT ONLY THE UPDATED MEMORY. NOTHING MORE.

{custom_message}{template_instructions}

## Task Description
{task_description}

## Current Task Trajectory
{task_trajectory}

## Current Memory

{current_memory}

## New Memory

"""

@dataclass
class IntrinsicMemoryDefault:
    """The update prompt every intrinsic memory module shares."""

    memory_update_prompt: str = MEMORY_UPDATE_PROMPT

INTRINSICMEMORY_DEFAULT: IntrinsicMemoryDefault = IntrinsicMemoryDefault()

#----------------------------------------------intrinsicmemory memory PDDL----------------------------------------------


MEMORY_SYSTEM_PROMPT_PDDL = """
You are a MEMORY UPDATER for a PDDL-style planning agent.

Your job:
- Maintain a compact JSON memory capturing stable, reusable information across tasks and domains.
- Only store information that improves future planning: common strategies, mistakes, valid action patterns, state-transition insights.
- Do not store long histories. Keep everything concise and deduplicated.

Inputs you receive each update call:
- current_memory: the previous memory as JSON (may be empty ⇒ re-init).
- latest_turn: the agent’s most recent Thought/Action/Observation.
- current_task: one of {blockworld, barman, gripper, tyreworld}.
- goal: current goal description.

OUTPUT:
- Return ONLY the updated memory as valid JSON following the template below.
- No extra commentary.

--------------------------
MEMORY TEMPLATE (ALWAYS FOLLOW)

{
  "task_summary": "brief description of PDDL planning setting",
  "global_strategies": [
    "high-level reusable planning heuristics across domains"
  ],
  "domains": {
    "blockworld": {
      "valid_action_patterns": ["pickup X", "putdown X", "stack X Y", "unstack X Y"],
      "good_strategies": ["free target block before stacking"],
      "invalid_patterns": ["wrong think format", "stack without clear base"],
      "mistakes": ["attempting pickup while arm full"]
    },
    "barman": {
      "valid_action_patterns": ["hand grasp glass", "fill-shot ...", "pour-shot-to-clean-shaker ..."],
      "good_strategies": ["ensure hand availability before filling"],
      "invalid_patterns": ["fill without holding glass"],
      "mistakes": ["grasp with occupied hand"]
    },
    "gripper": {
      "valid_action_patterns": ["move R1 R2", "pick O Room Gripper", "drop O Room Gripper"],
      "good_strategies": ["carry multiple items before moving rooms"],
      "invalid_patterns": ["drop object in wrong room"],
      "mistakes": ["pick while gripper full"]
    },
    "tyreworld": {
      "valid_action_patterns": ["open X", "fetch O C", "loosen N H", "jack-up H"],
      "good_strategies": ["open boot early to access tools"],
      "invalid_patterns": ["loosen nut without wrench"],
      "mistakes": ["inflate wheel without pump"]
    }
  },
  "tasks": [
    {
      "id": "identifier or hash of goal",
      "goal": "exact goal text",
      "status": "pending|solved",
      "helpful_observations": ["short state insights from valid steps"],
      "invalid_actions": ["summaries of failed attempts"],
      "progress_notes": ["short planning insights for this task"]
    }
  ]
}

--------------------------
UPDATE INSTRUCTIONS

1. Parse current_memory.  
   - If empty or invalid, initialize using the template above.

2. Update the domain-specific sections:
   - From latest_turn, add new useful action patterns, invalid patterns, or mistakes.
   - Keep lists short, deduplicated, and generalisable.

3. Update global_strategies if the latest_turn reveals a robust cross-domain heuristic.

4. Update the relevant task entry:
   - If no entry exists for this goal, create one.
   - Add helpful_observations if new actionable state insights appear.
   - Add invalid_actions if latest_turn shows an invalid move.
   - Add progress_notes for general reasoning improvements.
   - If task finished, mark status = "solved".

5. Return ONLY the updated JSON memory, nothing else.
"""

@dataclass
class IntrinsicMemoryPDDL:
    system_prompt: str = MEMORY_SYSTEM_PROMPT_PDDL

INTRINSICMEMORY_PDDL: IntrinsicMemoryPDDL = IntrinsicMemoryPDDL()
#----------------------------------------------intrinsicmemory memory FEVER----------------------------------------------


MEMORY_SYSTEM_PROMPT_FEVER = """
You are a MEMORY UPDATER for a question–answering agent.

The main agent:
- Solves fact-checking claims with interleaving Thought, Action, Observation steps.
- Actions: 
  - Search[entity]: search entity on Wikipedia, returns first paragraph or similar entities.
  - Lookup[keyword]: return next sentence containing keyword from last successful Search passage.
  - Finish[answer]: ends task with answer in {SUPPORTS, REFUTES, NOT ENOUGH INFO}.

Your job:
- Maintain a compact JSON memory capturing only stable, reusable information:
  - Recurrent strategies that work well.
  - Typical failure modes and how to avoid them.
  - Useful entities/queries/evidence patterns.
  - Final outcomes for claims and key reasons.
- Each time you are called, you receive:
  - current_memory: a JSON string (may be empty or invalid ⇒ re-init).
  - latest_turn: the agent’s latest Thought/Action/Observation block(s) and, if present, its final Finish[…].
  - current_claim: the claim currently being checked.

  MEMORY TEMPLATE (ALWAYS FOLLOW THIS SHAPE)

{
  "task_summary": "Short description of this QA + Wikipedia + REFUTES/SUPPORTS/NOT ENOUGH INFO setup.",
  "global_strategies": [
    "Reusable high-level strategies for searching, looking up, and deciding answers."
  ],
  "claims": [
    {
      "id": "short identifier for the claim if available, else a hash or index",
      "text": "exact or near-exact claim text",
      "status": "pending | answered",
      "final_answer": "SUPPORTS | REFUTES | NOT ENOUGH INFO | null",
      "key_entities": [
        "main entities (people, places, works, organizations, etc.)"
      ],
      "useful_search_queries": [
        "good Search[...] strings that led to helpful passages for this or similar claims"
      ],
      "supporting_evidence": [
        "very short paraphrased evidence snippets that support the claim"
      ],
      "refuting_evidence": [
        "very short paraphrased evidence snippets that refute the claim"
      ],
      "reasoning_notes": [
        "1–2 short notes on how the answer was decided or why it is still uncertain"
      ]
    }
  ],
  "tool_memory": {
    "search": {
      "good_patterns": [
        "e.g., include disambiguating info like (song), (film), or year"
      ],
      "bad_patterns": [
        "e.g., searching overly generic titles that return unrelated entities"
      ]
    },
    "lookup": {
      "good_patterns": [
        "e.g., use precise keywords like 'Billboard Hot 100', 'setting', 'born'"
      ],
      "bad_patterns": [
        "e.g., using keywords that appear in many irrelevant sentences"
      ]
    }
  },
  "mistakes_to_avoid": [
    "Stable lessons from past errors, e.g. 'Do not assume city vs. fictional town without checking setting sentence.'"
  ]
}

UPDATE INSTRUCTIONS (BE CONCISE)

1. Parse current_memory:
    - If empty or invalid: initialize a fresh JSON using the template above, filling minimal useful defaults.
    - Otherwise, keep the existing structure and keys; update values in place.
2. Read latest_turn and current_claim and decide what NEW, STABLE information to add or refine:
    - If a claim’s final Finish[ANSWER] appears:
        - Either create or update a claims entry for this claim (matching by id or text).
        - Set status to "answered" and final_answer to the chosen label.
        - Add at most 1–3 short supporting_evidence or refuting_evidence paraphrases derived from the Observations.
        - Add at most 1–2 reasoning_notes that capture the main decision logic or uncertainty.
    - If the claim is still in progress (no Finish yet):
        - Optionally create/update a claims entry with status "pending" and add key entities or useful queries discovered so far.
3. From latest_turn, update general patterns, but keep all lists short and non-duplicated:
- Add new, clearly useful search or lookup patterns to tool_memory.*.good_patterns.
- Add recurring unhelpful behaviors to tool_memory.*.bad_patterns.
- Add robust lessons to mistakes_to_avoid (only if likely to help future claims).
- Keep each string very short (aim ≤ 25 tokens) and do not re-add near-duplicates.
4. Maintain compactness:
- Prefer updating existing entries instead of creating many similar ones.
- If any list grows too long (e.g., > 15 items), you may drop the least useful or most specific ones.
- Never store raw long passages; only short paraphrased evidence.
5. Output format:
- Return ONLY the updated memory as valid JSON matching the template structure.
- Do NOT include explanations, markup, or any text outside the JSON.
"""


@dataclass
class IntrinsicMemoryFEVER:
    system_prompt: str = MEMORY_SYSTEM_PROMPT_FEVER

INTRINSICMEMORY_FEVER: IntrinsicMemoryFEVER = IntrinsicMemoryFEVER()
#----------------------------------------------intrinsicmemory memory ALFWORLD----------------------------------------------


MEMORY_SYSTEM_PROMPT_ALFWORLD = """
You are a MEMORY UPDATER for an Alfworld household agent.

Environment:
- The main agent solves tasks like locating, cleaning, heating/cooling, and placing objects.
- It MUST use only these command patterns (a, b = object/location vars):
  1) `take a from b.`
  2) `go to a.`
  3) `open a.`
  4) `put a in/on b.`  (never "in" alone, never "on" alone)
  5) `clean a with b.`
  6) `heat a with b.`
  7) `cool a with b.`
  8) `use a.`
  9) `think: xxx`

Your job:
- Maintain a compact JSON memory of reusable knowledge, not full histories.
- Capture: syntax constraints, search heuristics, object-location patterns, tool-usage patterns, and brief per-task notes.

Inputs each update:
- `current_memory`: JSON string (may be empty/invalid ⇒ re-init).
- `latest_turn`: the latest observation + thoughts + actions for a single step or short segment.
- `current_goal`: natural-language task description.
- `task_id`: short identifier for the current episode.

Output:
- A **single** valid JSON object following the template below.
- No extra text, comments, or formatting outside the JSON.

----------------
MEMORY TEMPLATE
----------------

{
  "task_summary": "Short description of Alfworld household tasks and allowed commands.",
  "syntax_rules": [
    "Allowed commands: take a from b, go to a, open a, put a in/on b, clean a with b, heat a with b, cool a with b, use a, think: xxx.",
    "Always use 'put a in/on b.'; never 'put a in b.' or 'put a on b.'.",
    "Every command must exactly match one allowed pattern, including punctuation."
  ],
  "global_strategies": [
    "First locate the needed object, then take it, then navigate to the target location/tool, then clean/heat/cool/place as required.",
    "Search likely locations first (e.g., food in fridge/diningtable/countertop; cleaning in sinkbasin; lamps on sidetable/desk/dresser)."
  ],
  "object_location_knowledge": {
    "apple": ["fridge", "diningtable", "garbagecan", "countertop", "sidemap"],
    "lettuce": ["fridge", "diningtable"],
    "soapbottle": ["countertop", "sinkbasin", "cabinet"],
    "soapbar": ["toilet", "sinkbasin", "shelf", "bathtubbasin"],
    "spraybottle": ["countertop", "cabinet"],
    "mug": ["countertop", "cabinet", "shelf", "fridge"],
    "pan": ["stoveburner", "countertop", "cabinet"],
    "potato": ["fridge", "diningtable", "garbagecan", "sinkbasin"],
    "creditcard": ["countertop", "dresser", "drawer"],
    "cellphone": ["coffeetable", "diningtable", "sofa", "dresser", "countertop"],
    "statue": ["dresser", "shelf", "coffeetable", "sidemap"],
    "desklamp": ["sidemap", "dresser", "desk", "diningtable"],
    "generic_defaults": ["cabinet", "drawer", "countertop", "shelf", "diningtable", "sidemap", "garbagecan"]
  },
  "tool_usage_patterns": {
    "clean": [
      "Find object, take it, go to sinkbasin X, clean object with sinkbasin X, then put object in/on target."
    ],
    "heat": [
      "Find object, take it, go to microwave X, heat object with microwave X, then put object in/on target."
    ],
    "cool": [
      "Find object, take it, go to fridge X, cool object with fridge X, then put object in/on target."
    ],
    "examine_with_light": [
      "Take object (e.g., bowl, pen, statue), then go to desklamp X location and use desklamp X."
    ],
    "put_two": [
      "Solve first instance fully (find, take, place), then search again for second instance and repeat."
    ]
  },
  "common_invalid_patterns": [
    "Using 'put a in b.' or 'put a on b.' instead of 'put a in/on b.'.",
    "Issuing a command not in the allowed list.",
    "Using malformed 'think' lines (must start with 'think: ' and then free text)."
  ],
  "tasks": [
    {
      "task_id": "short id or hash for an episode",
      "goal": "full goal text, e.g. 'put some apple in sidetable.'",
      "status": "pending or solved",
      "key_objects": [
        "names and indices of important objects, e.g. 'apple 3', 'sinkbasin 1', 'fridge 1'"
      ],
      "key_locations": [
        "locations actually used to solve or attempt the task, e.g. 'fridge 1', 'diningtable 1', 'sidemap 1'"
      ],
      "successful_action_patterns": [
        "Short general action schemas that worked, e.g. 'take X from Y; go to Z; put X in/on Z.'"
      ],
      "errors": [
        "Very short descriptions of mistakes made for this task, e.g. 'tried invalid put syntax', 'searched wrong room repeatedly'."
      ],
      "notes": [
        "1–3 short planning insights for this specific goal, e.g. 'apple often found in garbagecan if not on tables'."
      ]
    }
  ]
}

----------------
UPDATE INSTRUCTIONS
----------------

1. Parse `current_memory`.  
   - If empty/invalid, initialize a fresh object exactly following the template keys above, with minimal default values.

2. Update high-level knowledge from `latest_turn`:
   - If the environment response indicates a syntax error (e.g., invalid command or wrong 'in/on' usage), add a short, general rule to `common_invalid_patterns` or refine `syntax_rules`.
   - If an object is found in a location (e.g., "On the diningtable 1, you see a apple 3"), add/update that mapping in `object_location_knowledge` for the object type (deduplicate, keep list short and typical).
   - If a sequence like clean/heat/cool/examine or "put two" worked well, add or refine a single, short schema in `tool_usage_patterns` or `global_strategies`.

3. Update the `tasks` list:
   - Find the entry with matching `task_id`; if none exists, create a new one with `status` = "pending".
   - Set or update `goal` if needed from `current_goal`.
   - Append newly observed important objects/locations for this task to `key_objects` and `key_locations` (deduplicated).
   - When `latest_turn` shows a successful pattern (e.g., goal text appears satisfied or example episode ends with correct placement), add a generalized short description to `successful_action_patterns` and set `status` = "solved".
   - If `latest_turn` includes explicit feedback about invalid actions, add a brief description to this task’s `errors` and, if general, also to `common_invalid_patterns`.
   - Add at most 1–2 very short new items per update to `notes`, focusing on insights helpful for future similar goals.

4. Keep memory compact:
   - Avoid storing full transcripts or long sentences; paraphrase into short, reusable rules.
   - Deduplicate list entries by meaning; if a list grows too long (e.g., > 15 items), you may drop the least informative or most specific ones.

5. Output:
   - Return ONLY the updated memory JSON object, with all required top-level keys from the template and valid JSON syntax.
   - Do NOT output any explanations, comments, or text outside the JSON.
"""


@dataclass
class IntrinsicMemoryALFWORLD:
    system_prompt: str = MEMORY_SYSTEM_PROMPT_ALFWORLD

INTRINSICMEMORY_ALFWORLD: IntrinsicMemoryALFWORLD = IntrinsicMemoryALFWORLD()
#----------------------------------------------intrinsicmemory memory NO TEMPLATE----------------------------------------------


MEMORY_SYSTEM_PROMPT_GENERIC = """
You are an intelligent summarization agent. Your job is to update your current memory using your latest response with information that is useful for efficient task completion.
- Use only the information explicitly present in your prior responses.
- Do not invent or infer any information, actions, events, or observations that are not stated.
- Use clear and precise language. Avoid unnecessary details or storytelling.
- OUTPUT ONLY THE UPDATED MEMORY, NOTHING ELSE
"""


@dataclass
class IntrinsicMemoryNoTemplate:
    system_prompt: str = MEMORY_SYSTEM_PROMPT_GENERIC

INTRINSICMEMORY_NOTEMPLATE: IntrinsicMemoryNoTemplate = IntrinsicMemoryNoTemplate()

#----------------------------------------------intrinsicmemory LLM templated----------------------------------------------


MEMORY_TEMPLATE_SECTION = """

Use your latest response in the task trajectory to populate and update the current memory with factual information to solve the task 
based on the below instructions:
{template_instructions}"""

TEMPLATE_CREATION_PROMPT = """I have an AI agent that has to complete a task. 
The agent has a memory that is updated each time the LLM responds by comparing the latest response and the existing memory, 
and adding any new important information. The memory should be templated based on the nature of the task in a structured non-json format. 
The memory update is conducted as a prompted LLM call to update the memory. Provide the instructions to the agent for such an update operation, 
as well as the generic memory template for this particular task. Provide the full answer as a single prompt. 
Only include the most crucial details to the updating instructions to preserve token usage. 
Do not explain or describe the prompt, simply return the prompt and nothing more. This is the task description:
{task_description}
"""







@dataclass
class IntrinsicMemoryLLMTemplate:
    system_prompt: str = MEMORY_SYSTEM_PROMPT_GENERIC
    template_creation_prompt: str = TEMPLATE_CREATION_PROMPT
    memory_template_section: str = MEMORY_TEMPLATE_SECTION

INTRINSICMEMORY_LLM_TEMPLATE: IntrinsicMemoryLLMTemplate = IntrinsicMemoryLLMTemplate()
