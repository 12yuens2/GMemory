"""Which experiments a run covers, and how many at a time.

A flag given several values is swept: the experiments are the Cartesian product
of every such flag. One `experiment.run_experiment` call is one point in it, and
each one is isolated - a failure is recorded and the rest of the sweep continues.
"""

import argparse
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import product

from tqdm import tqdm

from mas.logging_utils import log_mas_to_console
from mas.module_map import MAS_MEMORY_MODULES
from mas.settings import LLMSettings, default_llm_settings

import results
from envs import ENVS
from experiment import run_experiment
from mas_workflow import MAS


def build_experiment_configs(args) -> list[dict]:
    """One config per combination of the flags given several values.

    Every parsed flag is read, so a flag added to the parser reaches the
    experiment without being named again here, and a flag declared `nargs='+'`
    is swept by that alone.
    """
    values = {
        name: list(value) if isinstance(value, (list, tuple)) else [value]
        for name, value in vars(args).items()
    }

    return [dict(zip(values, combination)) for combination in product(*values.values())]


def experiments_to_run(
    experiments: list[dict], overall_results_path: str
) -> tuple[list[dict], int]:
    """The experiments with no row in `overall_results_path` yet, and how many had one.

    Every writer appends, so re-running a killed sweep would add a second row for
    every experiment that had already finished, to be de-duplicated by hand
    before anything could be plotted.
    """
    recorded = results.recorded_experiments(overall_results_path)
    remaining = [
        experiment for experiment in experiments
        if results.experiment_key(experiment) not in recorded
    ]

    return remaining, len(experiments) - len(remaining)


def run_experiments(experiments: list[dict], num_workers: int) -> list[dict]:
    """Every experiment, in this process or in a pool of them.

    A pool is not spawned for a single experiment. The workers need nothing
    shared: they append to the same result files under a lock on each file, which
    is also what makes two separately submitted jobs safe.
    """
    if len(experiments) == 1 or num_workers <= 1:
        return [run_experiment(experiment_config) for experiment_config in experiments]

    ctx = multiprocessing.get_context('spawn')
    with ProcessPoolExecutor(
        max_workers=min(num_workers, len(experiments)), mp_context=ctx
    ) as executor:
        futures = [
            executor.submit(run_experiment, experiment_config)
            for experiment_config in experiments
        ]
        return [
            future.result()
            for future in tqdm(as_completed(futures), total=len(futures), desc='Running experiments')
        ]


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
    parser.add_argument('--max_tasks', type=int, default=None,
                        help="Override every task's configured max_tasks with this many tasks")

    # memory config
    parser.add_argument('--successful_topk', type=int, default=1, help='Number of successful trajs to be retrieved from memory.')
    parser.add_argument('--failed_topk', type=int, default=0, help='Number of failed trajs to be retrieved from memory.')
    parser.add_argument('--insights_topk', type=int, default=3, help='Number of insights to be retrieved from memory.')
    parser.add_argument('--threshold', type=float, default=0.0, help='threshold for traj similarity.')
    parser.add_argument('--use_projector', action='store_true', help='whether to use role projector.')
    parser.add_argument('--use_validator', action='store_true',
                        help='add a validator agent that checks the solver\'s action format before it is taken.')
    parser.add_argument('--hop', type=int, default=1, help='hop for traj similarity.')
    parser.add_argument('--intrinsic_cross_task', action='store_true',
                        help='keep an intrinsic memory across the tasks of a dataset instead of '
                             'starting each task from an empty one. No effect on the other memory '
                             'modules, which accumulate across tasks either way.')

    # experiment config
    parser.add_argument('--seed', type=int, nargs='+', default=[42], help='One or more seeds to run')
    parser.add_argument('--num_workers', type=int, default=num_cpus, help='Number of worker processes for parallel experiment execution.')
    parser.add_argument('--resume', action='store_true',
                        help='Skip the experiments already recorded in the overall results '
                             'file, instead of appending a second row for each of them.')

    # file paths
    parser.add_argument('--db_dir', type=str, default='./.db', help='Directory to store results, logs, and memory persistence for this run.')
    parser.add_argument('--overall_results_filename', type=str, default='overall_results.csv', help='Filename for overall results.')
    parser.add_argument('--failed_tasks_filename', type=str, default='failed_tasks.csv', help='Filename for failed tasks.')
    parser.add_argument('--failed_experiments_filename', type=str, default='failed_experiments.csv', help='Filename for failed experiments.')
    return parser


def main(argv: list[str] = None) -> None:
    args = build_arg_parser().parse_args(argv)

    settings: LLMSettings = default_llm_settings()
    print(f'LLM endpoint: {settings.api_base}, max_tokens: {settings.max_tokens}')

    if settings.log_responses:
        log_mas_to_console()

    overall_results_path = results.overall_results_path(
        args.db_dir, args.overall_results_filename
    )
    results.check_header(overall_results_path, results.AGGREGATE_COLUMNS)

    experiments = build_experiment_configs(args)
    if args.resume:
        experiments, already_done = experiments_to_run(experiments, overall_results_path)
        print(f'{already_done} experiments already recorded in {overall_results_path}, skipping')
        if not experiments:
            print('nothing left to run')
            return

    outcomes = run_experiments(experiments, args.num_workers)

    failed = [outcome for outcome in outcomes if outcome.get('status') == 'failed']
    if failed:
        # Reported by whichever worker wrote it, rather than reconstructed here
        failed_path = failed[0]['failed_path']
        print(f"\n{len(failed)}/{len(outcomes)} experiments failed. See {failed_path} for details.")
        for outcome in failed:
            print(f"  FAILED: task={outcome['task']} mas_memory={outcome['mas_memory']} seed={outcome['seed']} — {outcome['error']}")
