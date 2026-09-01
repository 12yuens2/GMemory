import json
import jsonlines

from mas.utils import repo_path

from .base_env import BaseEnv, BaseRecorder
from .alfworld_env import AlfworldEnv, AlfworldRecorder, get_env_name_from_gamefile, prefixes
from .sciworld_env import SciworldEnv, SciworldRecorder, build_simplification_str
from .fever_env import FeverEnv, FeverRecorder
from .pddl_env.pddl_env import PDDLEnv, PDDLRecorder, get_all_environment_configs

TASKS_PATH = {
    'alfworld': repo_path('data/alfworld/alfworld_tasks_suffix.json'),
    'fever': repo_path('data/fever/fever_dev.jsonl'),
    'pddl': repo_path('data/pddl/test.jsonl'),
    'sciworld': repo_path('data/sciworld/test.jsonl'),
}

PDDL_DOMAINS = ["barman", "blockworld", "gripper", "tyreworld"]


def load_alfworld_tasks() -> list[dict]:
    with open(TASKS_PATH['alfworld'], 'r') as reader:
        rows = json.load(reader)

    return [
        {
            'task': f'{row["goal"]}',
            'env_kwargs': {
                'config': 'alfworld',
                "gamefile": row["gamefile"],
            },
            'task_type': prefixes[get_env_name_from_gamefile(row["gamefile"])],
            'env_name': get_env_name_from_gamefile(row["gamefile"])
        } for row in rows
    ]


def load_sciworld_tasks() -> list[dict]:
    with open(TASKS_PATH['sciworld'], 'r', encoding='utf-8') as reader:
        return [
            {
                "id": item['id'],
                "task_name": item["additional_info"]["env_name"],
                "var": item["additional_info"]["var"],
                "modified_goal": item["goal"],
                "subgoals": item['subgoals'],
                "difficulty": item["difficulty"],
                "simplification_str": build_simplification_str()
            }
            for item in jsonlines.Reader(reader)
        ]


def load_fever_tasks() -> list[dict]:
    with open(TASKS_PATH['fever'], 'r') as reader:
        return [
            {
                'task': row['claim'],
                'answer': row['label'],
                'env_name': 'fever',
            }
            for row in (json.loads(line) for line in reader)
        ]


def load_pddl_tasks() -> list[dict]:
    return get_all_environment_configs(PDDL_DOMAINS, TASKS_PATH['pddl'])


TASK_LOADERS = {
    'alfworld': load_alfworld_tasks,
    'sciworld': load_sciworld_tasks,
    'fever': load_fever_tasks,
    'pddl': load_pddl_tasks,
}

ENVS = {
    'alfworld': AlfworldEnv,
    'sciworld': SciworldEnv,
    'fever': FeverEnv,
    'pddl': PDDLEnv
}

RECORDERS = {
    'alfworld': AlfworldRecorder,
    'sciworld': SciworldRecorder,
    'fever': FeverRecorder,
    'pddl': PDDLRecorder
}

_datasets: dict[str, list[dict]] = {}


def get_env(task: str, env_config: dict, max_trials: int) -> BaseEnv:
    
    if ENVS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return ENVS.get(task)(env_config, max_trials)

def get_recorder(task: str, working_dir: str, namespace: str) -> BaseRecorder:
    
    if RECORDERS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')
    
    return RECORDERS.get(task)(working_dir=working_dir, namespace=namespace)

def get_task(task: str, max_tasks: int = None) -> list[dict]:
    """The dataset for `task`, parsed on first use and kept for the next call.

    `max_tasks` takes the first n tasks. It is per-task configuration, in
    tasks/configs.yaml - not a sample, so two runs of the same task cover the
    same tasks.
    """
    if TASK_LOADERS.get(task) is None:
        raise ValueError(f'Unsupported task type: {task}')

    if task not in _datasets:
        _datasets[task] = TASK_LOADERS[task]()

    dataset = _datasets[task]
    return dataset[:max_tasks] if max_tasks is not None else dataset
