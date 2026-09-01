import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import multiprocessing
import traceback
import yaml
from dataclasses import dataclass, field
import argparse
import random
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product
from tqdm import tqdm

from mas.module_map import MAS_MEMORY_MODULES, module_map
from mas.reasoning import ReasoningBase
from mas.memory import MASMemoryBase
from mas.llm import LLMCallable, GPTChat, TokenTracker
from mas.settings import LLMSettings, default_llm_settings
from mas.mas import MetaMAS
from mas.utils import EmbeddingFunc

import results
from envs import ENVS, BaseEnv, BaseRecorder, get_env, get_recorder, get_task
from mas_workflow import MAS, get_mas
from prompts import get_dataset_system_prompt, get_task_few_shots
from utils import get_model_type

with open('tasks/configs.yaml') as reader:
    CONFIG: dict = yaml.safe_load(reader)


@dataclass
class TaskManager:
    task_name: str              # task name
    mas_type: str               # type of mas
    memory_type: str            # memory type
    tasks: list[dict]           # all tasks
    env: BaseEnv                # interative datatset environment
    recorder: BaseRecorder      # record experiment results
    mas: MetaMAS                # multi-agent system
    seed: int = None            # this experiment's seed
    model: str = None           # the LLM this experiment queries
    mas_config: dict = field(default_factory=dict)   # mas configs
    mem_config: dict = field(default_factory=dict)   # memory configs
    token_tracker: TokenTracker = None   # token accounting for this experiment's LLM calls

    def identity(self) -> dict:
        """Which experiment a result row belongs to."""
        return results.identity(
            model=self.model,
            task=self.task_name,
            mas_type=self.mas_type,
            mas_memory=self.memory_type,
            use_validator=self.mas_config.get('use_validator', False),
        )


def trial_budget(task: str, override: int = None) -> int:
    """Trials one episode of `task` gets: its `max_steps`, or `override`."""
    if override is not None:
        return override

    budget = CONFIG.get(task, {}).get('max_steps')
    if budget is None:
        raise KeyError(f"'{task}' has no max_steps in tasks/configs.yaml")
    return budget


def build_task(
    task: str,
    mas_type: str,
    memory_type: str,
    seed: int,
    working_dir: str,
    model: str = None,
    max_trials: int = None,
) -> TaskManager:

    with open(CONFIG.get(task).get('env_config_path')) as reader:
        config = yaml.safe_load(reader)

    env: BaseEnv = get_env(task, config, trial_budget(task, max_trials))
    recorder: BaseRecorder = get_recorder(task, working_dir=working_dir, namespace=f'total_task-seed_{seed}')
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
        seed=seed,
        model=model,
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

def run_task(
    task_manager: TaskManager,
    working_dir: str,
    output_lock=None,
) -> None:
    task_manager.recorder.dataset_begin()
    progress_path = results.progress_path(working_dir, task_manager.task_name, task_manager.memory_type)

    failed_tasks: list[dict] = []

    for task_id, task_config in tqdm(enumerate(task_manager.tasks), total=len(task_manager.tasks), desc="Running Tasks"):
        try:
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

            episode = task_manager.mas.schedule(task_config) # Schedule method from the mas_workflow (e.g. autogen)
            task_manager.recorder.task_end(episode)
        except Exception as error:
            failed_tasks.append({'task_id': task_id, 'error': error})
            task_manager.recorder.log(
                f'TASK FAILED task_id={task_id}: {type(error).__name__}: {error}\n'
                f'{traceback.format_exc()}'
            )
            print(
                f'Task failed, continuing: task={task_manager.task_name} task_id={task_id} '
                f'{type(error).__name__}: {error}',
                file=sys.stderr,
            )
            continue

        tracker = task_manager.token_tracker
        completion_tokens, prompt_tokens = tracker.completion_tokens, tracker.prompt_tokens
        intrinsic_completion_tokens = tracker.intrinsic_completion_tokens
        intrinsic_prompt_tokens = tracker.intrinsic_prompt_tokens
        task_manager.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}\n')
        task_manager.recorder.log(f'intrinsic completion tokens:{intrinsic_completion_tokens}, intrinsic_prompt_tokens:{intrinsic_prompt_tokens}\n')
        task_manager.recorder.log(f'seed: {task_manager.seed}\n')

        # a partial result per completed task, so a killed job keeps what it measured
        print(results.write_row(
            progress_path,
            results.RESULT_COLUMNS,
            results.result_row(
                identity_fields=task_manager.identity(),
                seed=task_manager.seed,
                max_trials=task_manager.env.max_trials,
                averages=task_manager.recorder.average_results(),
                tracker=tracker,
            ),
            output_lock=output_lock,
        ))

    if failed_tasks:
        summary = (
            f'{len(failed_tasks)}/{len(task_manager.tasks)} tasks failed and are excluded '
            f'from the averages above'
        )
        task_manager.recorder.log(summary)
        print(summary, file=sys.stderr)
        _write_failed_tasks(task_manager, failed_tasks, working_dir, output_lock=output_lock)

    task_manager.recorder.dataset_end()


def _write_failed_tasks(
    task_manager: TaskManager,
    failed_tasks: list[dict],
    working_dir: str,
    output_lock=None,
    filename: str = 'failed_tasks.csv',
) -> None:
    """One row per task that could not be run, alongside the experiment's results."""
    path = os.path.join(working_dir, filename)

    for failure in failed_tasks:
        results.write_row(
            path,
            results.FAILED_TASK_COLUMNS,
            {
                **task_manager.identity(),
                'task_id': failure['task_id'],
                'seed': task_manager.seed,
                **results.failure_fields(failure['error']),
            },
            output_lock=output_lock,
        )


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
        'use_validator': [args.use_validator],
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
    use_validator = experiment_config['use_validator']
    hop = experiment_config['hop']
    db_dir = experiment_config['db_dir']

    # set save dirs
    working_dir = os.path.join(db_dir, get_model_type(model_type), task_name, mas_type, f'{mas_memory_type}')
    os.makedirs(working_dir, exist_ok=True)

    try:
        random.seed(seed)

        task_configs: TaskManager = build_task(
            task_name, mas_type, mas_memory_type, seed, working_dir,
            model=model_type, max_trials=max_trials,
        )
        task_configs.mas_config['successful_topk'] = successful_topk
        task_configs.mas_config['failed_topk'] = failed_topk
        task_configs.mas_config['insights_topk'] = insights_topk
        task_configs.mas_config['threshold'] = threshold
        task_configs.mas_config['use_projector'] = use_projector
        task_configs.mas_config['use_validator'] = use_validator

        # each seed gets its own memory persistence dir so concurrent seeds of the same
        # experiment config never read/write the same graph/vector-store/insights files
        memory_dir = os.path.join(working_dir, f'seed_{seed}')
        task_configs.mem_config.update(
            working_dir=memory_dir,
            hop=hop
        )

        build_mas(task_configs, reasoning_type, mas_memory_type, model_type)
        run_task(task_configs, working_dir, output_lock=output_lock)

        tracker = task_configs.token_tracker
        completion_tokens, prompt_tokens = tracker.completion_tokens, tracker.prompt_tokens
        intrinsic_completion_tokens = tracker.intrinsic_completion_tokens
        intrinsic_prompt_tokens = tracker.intrinsic_prompt_tokens
        task_configs.recorder.log(f'completion_tokens:{completion_tokens}, prompt_tokens:{prompt_tokens}')
        task_configs.recorder.log(f'intrinsic completion tokens:{intrinsic_completion_tokens}, intrinsic_prompt_tokens:{intrinsic_prompt_tokens}')

        row = results.result_row(
            identity_fields=task_configs.identity(),
            seed=seed,
            max_trials=task_configs.env.max_trials,
            averages=task_configs.recorder.average_results(),
            tracker=tracker,
        )
        result_line = results.write_row(
            results.results_path(working_dir), results.RESULT_COLUMNS, row, output_lock=output_lock
        )
        print(result_line)

        results.write_row(
            results.overall_results_path(db_dir), results.RESULT_COLUMNS, row, output_lock=output_lock
        )

        return {
            'task': task_name,
            'mas_type': mas_type,
            'mas_memory': mas_memory_type,
            'seed': seed,
            'result_string': result_line,
            'status': 'success',
        }
    except Exception as e:
        print(f"Experiment failed: task={task_name} mas_memory={mas_memory_type} seed={seed}\n{traceback.format_exc()}", file=sys.stderr)
        failed_path = _write_failed_experiment(experiment_config, e, db_dir, output_lock=output_lock)
        return {
            'task': task_name,
            'mas_type': mas_type,
            'mas_memory': mas_memory_type,
            'seed': seed,
            'status': 'failed',
            'error': f'{type(e).__name__}: {e}',
            'failed_path': failed_path,
        }


def _write_failed_experiment(
    experiment_config: dict,
    error: Exception,
    db_dir: str,
    output_lock=None,
    filename: str = 'failed_experiments.csv',
) -> str:
    failed_path = os.path.join(db_dir, filename)
    results.write_row(
        failed_path,
        results.FAILED_EXPERIMENT_COLUMNS,
        {
            **results.identity(
                model=experiment_config.get('model', ''),
                task=experiment_config.get('task', ''),
                mas_type=experiment_config.get('mas_type', ''),
                mas_memory=experiment_config.get('mas_memory', ''),
                use_validator=experiment_config.get('use_validator', False),
            ),
            'seed': experiment_config.get('seed', ''),
            **results.failure_fields(error),
        },
        output_lock=output_lock,
    )
    return failed_path


def build_arg_parser() -> argparse.ArgumentParser:
    num_cpus = max(1, os.cpu_count() - 32)

    parser = argparse.ArgumentParser(description='Run tasks with specified modules.')
    parser.add_argument('--task', type=str, nargs='+', choices=sorted(ENVS), default=['alfworld'], help='One or more tasks to run')
    parser.add_argument('--mas_type', type=str, choices=sorted(MAS), required=True, help='Multi-agent workflow to run')
    parser.add_argument('--mas_memory', type=str, nargs='+', choices=sorted(MAS_MEMORY_MODULES), required=True, help='One or more mas memory modules to run')
    parser.add_argument('--reasoning', type=str, default='io', help='Specify reasoning module')
    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Specify the LLM model type')
    parser.add_argument('--max_trials', type=int, default=None,
                        help="Override every task's configured max_steps with this trial budget")
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--use_validator', action='store_true',
                        help='add a validator agent that checks the solver\'s action format before it is taken.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='One or more seeds to run')
    parser.add_argument('--num_workers', type=int, default=num_cpus, help='Number of worker processes for parallel experiment execution.')
    parser.add_argument('--db_dir', type=str, default='./.db', help='Directory to store results, logs, and memory persistence for this run.')
    return parser


if __name__ == '__main__':
    args = build_arg_parser().parse_args()

    # Resolved before any worker is spawned, so missing credentials are reported
    # once, here, rather than inside each worker's traceback.
    settings: LLMSettings = default_llm_settings()
    print(f'LLM endpoint: {settings.api_base}, max_tokens: {settings.max_tokens}')

    experiments = build_experiment_configs(args)
    if len(experiments) == 1 or args.num_workers <= 1:
        results = [run_experiment(experiment_config) for experiment_config in experiments]
    else:
        ctx = multiprocessing.get_context('spawn')
        with multiprocessing.Manager() as manager:
            output_lock = manager.Lock()
            with ProcessPoolExecutor(max_workers=min(args.num_workers, len(experiments)), mp_context=ctx) as executor:
                futures = [executor.submit(run_experiment, experiment_config, output_lock) for experiment_config in experiments]
                results = [future.result() for future in tqdm(as_completed(futures), total=len(futures), desc='Running experiments')]

    failed = [r for r in results if r.get('status') == 'failed']
    if failed:
        # Reported by whichever worker wrote it, rather than reconstructed here
        failed_path = failed[0]['failed_path']
        print(f"\n{len(failed)}/{len(results)} experiments failed. See {failed_path} for details.")
        for r in failed:
            print(f"  FAILED: task={r['task']} mas_memory={r['mas_memory']} seed={r['seed']} — {r['error']}")

