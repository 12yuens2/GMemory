import json
import os
import jsonlines

from .base_env import BaseEnv, BaseRecorder
from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes
from .sciworld_env import SciworldEnv, SciworldRecorder, build_simplification_str
from .fever_env import FeverEnv, FeverRecorder
from .pddl_env.pddl_env import PDDLEnv, PDDLRecorder, get_all_environment_configs
from .livecode_env import (
    LCBCodegenEnv, LCBCodegenRecorder,
    LCBCodeExecEnv, LCBCodeExecRecorder,
    LCBTestPredEnv, LCBTestPredRecorder,
    LCBSelfRepairEnv, LCBSelfRepairRecorder,
)

TASKS_PATH = {
    'alfworld': 'data/alfworld/alfworld_tasks_suffix.json',
    'fever': 'data/fever/fever_dev.jsonl',
    'pddl': 'data/pddl/test.jsonl',
    'sciworld': 'data/sciworld/test.jsonl',
    'lcb_codegen': 'data/lcb/codegen.json',
    'lcb_codeexec': 'data/lcb/codeexec.json',
    'lcb_testpred': 'data/lcb/testpred.json',
}

## Tasks
alfworld_tasks: list[dict] = [
    {
        'task': f'{row["goal"]}',
        'env_kwargs': {
            'config': 'alfworld',
            "gamefile": row["gamefile"],
        },
        'task_type': prefixes[get_env_name_from_gamefile(row["gamefile"])],
        'env_name': get_env_name_from_gamefile(row["gamefile"])
    } for row in json.load(open(TASKS_PATH['alfworld'], "r")) 
]

sciworld_tasks: list = []
with open(TASKS_PATH['sciworld'], 'r+', encoding='utf-8') as f:
    for item in jsonlines.Reader(f):
        task_name = item["additional_info"]["env_name"]
        var = item["additional_info"]["var"]
        sciworld_tasks.append({
            "id": item['id'],
            "task_name": task_name,
            "var": var,
            "modified_goal": item["goal"],
            "subgoals": item['subgoals'],
            "difficulty": item["difficulty"],
            "simplification_str": build_simplification_str()
        })

with open(TASKS_PATH['fever'], 'r') as f:
    fever_tasks = [
        {
            'task': row['claim'],
            'answer': row['label'],
            'env_name': 'fever',
        }
        for row in (json.loads(line) for line in f) 
    ][:100]


TASK_NAMES = ["barman", "blockworld", "gripper", "tyreworld"]
pddl_tasks: list[dict] = get_all_environment_configs(TASK_NAMES, TASKS_PATH['pddl'])


def _load_json(path: str) -> list[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


lcb_codegen_tasks: list[dict] = _load_json(TASKS_PATH['lcb_codegen'])
lcb_codeexec_tasks: list[dict] = _load_json(TASKS_PATH['lcb_codeexec'])
lcb_testpred_tasks: list[dict] = _load_json(TASKS_PATH['lcb_testpred'])


TASK_DATA = {
    'alfworld': alfworld_tasks,
    'sciworld': sciworld_tasks,
    'fever': fever_tasks,
    'pddl': pddl_tasks,
    'lcb_codegen': lcb_codegen_tasks,
    'lcb_codeexec': lcb_codeexec_tasks,
    'lcb_testpred': lcb_testpred_tasks,
}

ENVS = {
    'alfworld': AlfworldEnv,
    'sciworld': SciworldEnv,
    'fever': FeverEnv,
    'pddl': PDDLEnv,
    'lcb_codegen': LCBCodegenEnv,
    'lcb_codeexec': LCBCodeExecEnv,
    'lcb_testpred': LCBTestPredEnv,
    'lcb_selfrepair': LCBSelfRepairEnv,
}

RECORDERS = {
    'alfworld': AlfworldRecorder,
    'sciworld': SciworldRecorder,
    'fever': FeverRecorder,
    'pddl': PDDLRecorder,
    'lcb_codegen': LCBCodegenRecorder,
    'lcb_codeexec': LCBCodeExecRecorder,
    'lcb_testpred': LCBTestPredRecorder,
    'lcb_selfrepair': LCBSelfRepairRecorder,
}


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    
    if ENVS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return ENVS.get(task)(env_config, max_trials)

def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    
    if RECORDERS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return RECORDERS.get(task)(working_dir=working_dir, namespace=namespace)

def get_task(task: str, **kwargs) -> list[dict]:

    if task == 'lcb_selfrepair':
        codegen_results_path = kwargs.get('codegen_results_path')
        if not codegen_results_path or not os.path.exists(codegen_results_path):
            raise FileNotFoundError(
                f"lcb_selfrepair: results file not found at '{codegen_results_path}'.\n"
                "Run lcb_codegen first, then pass --codegen_results <path/to/results.json>"
            )
        with open(codegen_results_path) as f:
            results = json.load(f)
        failed = [r for r in results if not r.get('passed', True)]
        if not failed:
            raise ValueError("No failed tasks found in codegen results file. Nothing to repair.")
        return failed

    if TASK_DATA.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')

    return TASK_DATA.get(task)