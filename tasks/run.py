import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import shutil
import multiprocessing
import yaml
from dataclasses import dataclass, field
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm

import mas
from mas.agents import Agent
from mas.module_map import module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, GPTChat, TokenTracker
from mas.mas import MetaMAS
from mas.utils import EmbeddingFunc

from envs import BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type

with open('tasks/configs.yaml') as reader:
    CONFIG: dict = yaml.safe_load(reader)

WORKING_DIR: str = None
DEFAULT_DB_DIR: str = './.db'

@dataclass
class TaskManager:
    task_name: str              # task name
    mas_type: str               # type of mas
    memory_type: str            # memory type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    mas: MetaMAS                # multi-agent system
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs
    token_tracker: TokenTracker = None   # token accounting for this experiment's LLM calls


def build_task(task: str, mas_type: str, memory_type: str, max_steps: int, seed: int) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)

    env: BaseEnv = get_env(task, config, max_steps)
    recorder: BaseRecorder = get_recorder(task, working_dir=WORKING_DIR, namespace=f'total_task-seed_{seed}')
    tasks: list[dict] = get_task(task)
    mas_workflow: MetaMAS = get_mas(mas_type)
    mas_config: dict = CONFIG.get(mas_type, {})

    return TaskManager(
        task_name=task,
        mas_type=mas_type,
        memory_type=memory_type,
        tasks=tasks,
        env=env,
        recorder=recorder,
        mas=mas_workflow,
        mas_config=mas_config
    )

def build_mas(
    task_manager: TaskManager,
    reasoning: str = None,
    mas_memory: str = None,
    llm_type: str = None,
) -> None:
    
    embed_func = EmbeddingFunc(CONFIG.get('embedding_model', "sentence-transformers/all-MiniLM-L6-v2")) 
    reasoning_module_type, mas_memory_module_type = module_map(reasoning, mas_memory)

    llm_model: LLMCallable = GPTChat(model_name=llm_type)
    task_manager.token_tracker = llm_model.tracker
    reasoning_module: ReasoningBase = reasoning_module_type(llm_model=llm_model)
    mas_memory_module: MASMemoryBase = mas_memory_module_type(
        namespace=mas_memory,
        global_config=task_manager.mem_config,
        llm_model=llm_model,
        embedding_func=embed_func 
    )
    task_manager.mas.add_observer(task_manager.recorder)  
    task_manager.mas.build_system(reasoning_module, mas_memory_module, task_manager.env, task_manager.mas_config)

def append_local_result(output_path: str, result_string: str, output_lock=None) -> None:
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    if output_lock is not None:
        with output_lock:
            with open(output_path, 'a', encoding='utf-8') as writer:
                writer.write(result_string)
    else:
        with open(output_path, 'a', encoding='utf-8') as writer:
            writer.write(result_string)


def run_task(
    task_manager: TaskManager,
    seed: int,
    model_type: str = None,
    task_name: str = None,
    mas_memory_type: str = None,
    output_lock=None,
) -> None:
    task_manager.recorder.dataset_begin()
    result_path = os.path.join(WORKING_DIR, f"{task_manager.task_name}-{task_manager.memory_type}-results.csv")

    for task_id, task_config in tqdm(enumerate(task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        task_manager.recorder.task_begin(task_id, task_config)

        task_main, task_description = task_manager.mas.env.set_env(task_config)
        few_shots: list[str] = get_task_few_shots(
            dataset=task_manager.task_name,
            task_config=task_config,
            few_shots_num=CONFIG.get(task_manager.task_name).get('few_shots_num', 0)
        )
        task_config.update(task_main=task_main, task_description=task_description, few_shots=few_shots)

        task_instruction: str = get_dataset_system_prompt(task_manager.task_name, task_config=task_config)
        for agent in task_manager.mas.agents_team.values():
            task_manager.recorder.log(f'------------ MAS Agent: {agent.name} ------------')
            task_manager.recorder.log(agent.add_task_instruction(task_instruction))

        reward, done, trials = task_manager.mas.schedule(task_config) # Schedule method from the mas_workflow (e.g. autogen)
        task_manager.recorder.task_end(reward, done, trials)

        tracker = task_manager.token_tracker
        completion_tokens, prompt_tokens = tracker.completion_tokens, tracker.prompt_tokens
        intrinsic_completion_tokens = tracker.intrinsic_completion_tokens
        intrinsic_prompt_tokens = tracker.intrinsic_prompt_tokens
        task_manager.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}\n')
        task_manager.recorder.log(f'intrinsic completion tokens:{intrinsic_completion_tokens}, intrinsic_prompt_tokens:{intrinsic_prompt_tokens}\n')
        task_manager.recorder.log(f'seed: {seed}\n')

        # output results as each task completes
        results, dones, trials = task_manager.recorder.average_results()
        result_string = f"{model_type},{task_name},{mas_memory_type},{seed},{results},{dones},{trials},{completion_tokens},{prompt_tokens},{intrinsic_completion_tokens},{intrinsic_prompt_tokens}\n"
        print(result_string)

        append_local_result(result_path, result_string, output_lock=output_lock)

    task_manager.recorder.dataset_end()


def build_experiment_configs(args) -> list[dict]:
    params = {
        'task': getattr(args, 'task'),
        'mas_type': getattr(args, 'mas_type'),
        'mas_memory': getattr(args, 'mas_memory'),
        'reasoning': getattr(args, 'reasoning'),
        'model': getattr(args, 'model'),
        'seed': getattr(args, 'seed'),
        'max_trials': [args.max_trials],
        'successful_topk': [args.successful_topk],
        'failed_topk': [args.failed_topk],
        'insights_topk': [args.insights_topk],
        'threshold': [args.threshold],
        'use_projector': [args.use_projector],
        'hop': [args.hop],
        'num_workers': [args.num_workers],
        'db_dir': [args.db_dir],
    }

    values = {key: value if isinstance(value, (list, tuple)) else [value] for key, value in params.items()}
    keys = [key for key, value in values.items() if len(value) > 1]
    if not keys:
        return [{key: value[0] for key, value in values.items()}]

    configs = []
    for combination in product(*[values[key] for key in params.keys()]):
        configs.append(dict(zip(params.keys(), combination)))
    return configs


def run_experiment(experiment_config: dict, output_lock=None) -> dict:
    task_name = experiment_config['task']
    mas_type = experiment_config['mas_type']
    mas_memory_type = experiment_config['mas_memory']
    reasoning_type = experiment_config['reasoning']
    model_type = experiment_config['model']
    max_trials = experiment_config['max_trials']
    seed = experiment_config['seed']
    successful_topk = experiment_config['successful_topk']
    failed_topk = experiment_config['failed_topk']
    insights_topk = experiment_config['insights_topk']
    threshold = experiment_config['threshold']
    use_projector = experiment_config['use_projector']
    hop = experiment_config['hop']
    db_dir = experiment_config['db_dir']

    # set save dirs
    global WORKING_DIR
    WORKING_DIR = os.path.join(db_dir, get_model_type(model_type), task_name, mas_type, f'{mas_memory_type}')
    os.makedirs(WORKING_DIR, exist_ok=True)

    random.seed(seed)

    task_configs: TaskManager = build_task(task_name, mas_type, mas_memory_type, max_trials, seed)
    task_configs.mas_config['successful_topk'] = successful_topk
    task_configs.mas_config['failed_topk'] = failed_topk
    task_configs.mas_config['insights_topk'] = insights_topk
    task_configs.mas_config['threshold'] = threshold
    task_configs.mas_config['use_projector'] = use_projector

    # each seed gets its own memory persistence dir so concurrent seeds of the same
    # experiment config never read/write the same graph/vector-store/insights files
    memory_dir = os.path.join(WORKING_DIR, f'seed_{seed}')
    task_configs.mem_config.update(
        working_dir=memory_dir,
        hop=hop
    )

    build_mas(task_configs, reasoning_type, mas_memory_type, model_type)
    run_task(
        task_configs,
        seed,
        model_type=model_type,
        task_name=task_name,
        mas_memory_type=mas_memory_type,
        output_lock=output_lock,
    )

    tracker = task_configs.token_tracker
    completion_tokens, prompt_tokens = tracker.completion_tokens, tracker.prompt_tokens
    intrinsic_completion_tokens = tracker.intrinsic_completion_tokens
    intrinsic_prompt_tokens = tracker.intrinsic_prompt_tokens
    task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}, price={completion_tokens*15/1000000+prompt_tokens*5/1000000}')
    task_configs.recorder.log(f'intrinsic completion tokens:{intrinsic_completion_tokens}, intrinsic_prompt_tokens:{intrinsic_prompt_tokens}')

    results, dones, trials = task_configs.recorder.average_results()
    result_string = f"{model_type},{task_name},{mas_memory_type},{results},{dones},{trials},{completion_tokens},{prompt_tokens},{intrinsic_completion_tokens},{intrinsic_prompt_tokens},{seed}\n"
    print(result_string)

    append_local_result(os.path.join(WORKING_DIR, 'results.csv'), result_string, output_lock=output_lock)
    _write_overall_result(experiment_config, result_string, db_dir, output_lock=output_lock)

    return {
        'task': task_name,
        'mas_type': mas_type,
        'mas_memory': mas_memory_type,
        'seed': seed,
        'result_string': result_string,
    }


def _write_overall_result(experiment_config: dict, result_string: str, db_dir: str, output_lock=None) -> None:
    headers = [
        'task',
        'mas_type',
        'mas_memory',
        'model',
        'seed',
        'max_trials',
        'num_workers',
        'results',
        'dones',
        'trials',
        'completion_tokens',
        'prompt_tokens',
        'intrinsic_completion_tokens',
        'intrinsic_prompt_tokens',
    ]

    basic_values = [
        str(experiment_config.get('task', '')),
        str(experiment_config.get('mas_type', '')),
        str(experiment_config.get('mas_memory', '')),
        str(experiment_config.get('model', '')),
        str(experiment_config.get('seed', '')),
        str(experiment_config.get('max_trials', '')),
        str(experiment_config.get('num_workers', '')),
    ]

    result_fields = result_string.strip().split(',')
    if len(result_fields) >= 10:
        summary_fields = result_fields[3:10]
    else:
        summary_fields = result_fields[3:]

    overall_line = ','.join(basic_values + summary_fields) + '\n'
    overall_results_path = os.path.join(db_dir, 'overall_results.csv')

    if output_lock is not None:
        with output_lock:
            if not os.path.exists(overall_results_path) or os.path.getsize(overall_results_path) == 0:
                header_line = ','.join(headers) + '\n'
                append_local_result(overall_results_path, header_line, output_lock=None)
            append_local_result(overall_results_path, overall_line, output_lock=None)
    else:
        if not os.path.exists(overall_results_path) or os.path.getsize(overall_results_path) == 0:
            header_line = ','.join(headers) + '\n'
            append_local_result(overall_results_path, header_line, output_lock=None)
        append_local_result(overall_results_path, overall_line, output_lock=None)


if __name__ == '__main__':
    # settings
    num_cpus = max(1, os.cpu_count() - 32)

    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument('--task', type=str, nargs='+', choices=['alfworld', 'fever', 'pddl', 'sciworld'], default=['alfworld'], help='One or more tasks to run')
    parser.add_argument('--mas_type', type=str, choices=['autogen','autogen_mas', 'macnet', 'dylan'])
    parser.add_argument('--mas_memory', type=str, nargs='+', default=['none'], help='One or more mas memory modules to run')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=30, help='max number of steps')
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='One or more seeds to run')
    parser.add_argument('--num_workers', type=int, default=num_cpus, help='Number of worker processes for parallel experiment execution.')
    parser.add_argument('--db_dir', type=str, default=DEFAULT_DB_DIR, help='Directory to store results, logs, and memory persistence for this run.')

    args = parser.parse_args()

    experiments = build_experiment_configs(args)
    if len(experiments) == 1 or args.num_workers <= 1:
        for experiment_config in experiments:
            run_experiment(experiment_config)
    else:
        ctx = multiprocessing.get_context('spawn')
        with multiprocessing.Manager() as manager:
            output_lock = manager.Lock()
            with ProcessPoolExecutor(max_workers=min(args.num_workers, len(experiments)), mp_context=ctx) as executor:
                futures = [executor.submit(run_experiment, experiment_config, output_lock) for experiment_config in experiments]
                for future in tqdm(as_completed(futures), total=len(futures), desc='Running experiments'):
                    future.result()

