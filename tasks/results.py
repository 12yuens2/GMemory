"""The result rows an experiment writes, and the CSV files they go in.

Three rules hold across every file here: the same identity columns come first,
`seed` is last, and wherever token counts appear they appear in the same order.
A reader can then line up a row from any of them, and nothing has to recover a
field by its position.

Four files, and what one row of each is:

    <task>-<memory>-task_results.csv      one completed task, raw
    <task>-<memory>-seed_<n>-progress.csv one completed task, as means so far
    overall_results.csv                   one finished experiment
    failed_tasks.csv                      one task that could not be run
    failed_experiments.csv                one experiment that could not be run

The task file is the raw material: one row per task carrying that episode's own
reward, done and trials, and the tokens that episode spent, so anything the mean
hides - variance, which tasks were solved, cost per task - can be computed from
it afterwards.

The progress file is the same tasks reported as running means, and exists only so
a job the scheduler kills part-way through a dataset leaves something readable
without processing. It is removed once that experiment's `overall_results.csv`
row is written, since from then on it says nothing the other two files do not.
It is per seed, so concurrent seeds of one config do not share the file one of
them will delete.

Two things worth knowing about the token columns. On a task row they are that
episode's spend, and they will not sum to the run total when a task failed
part-way, because a task with no row still spent what it spent. On an aggregate
row they are the run's cumulative total.

Each filename is an argument with a default, so a caller writing somewhere else
does not have to know how the name is built.
"""

import csv
import io
import os
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

from mas.llm import TokenTracker

if TYPE_CHECKING:
    from mas.mas import EpisodeResult
    from tasks.envs.base_env import AggregateResults


def _tokens(tracker: TokenTracker) -> dict:
    return {
        'completion_tokens': tracker.completion_tokens,
        'prompt_tokens': tracker.prompt_tokens,
        'intrinsic_completion_tokens': tracker.intrinsic_completion_tokens,
        'intrinsic_prompt_tokens': tracker.intrinsic_prompt_tokens,
    }


@dataclass(frozen=True)
class Measurements:
    """Means over episodes, for the files reporting an aggregate.

    `tasks_scored` is how many episodes the means are over. Without it a reader
    cannot tell a mean over 200 tasks from a mean over 3, which is also how a
    partial progress row reports its progress.
    """

    mean_reward: float
    mean_done: float
    mean_trials: float
    tasks_scored: int
    completion_tokens: int
    prompt_tokens: int
    intrinsic_completion_tokens: int
    intrinsic_prompt_tokens: int

    @classmethod
    def of(cls, averages: "AggregateResults", tracker: TokenTracker) -> "Measurements":
        return cls(
            mean_reward=averages.mean_reward,
            mean_done=averages.mean_done,
            mean_trials=averages.mean_trials,
            tasks_scored=averages.episode_count,
            **_tokens(tracker),
        )


@dataclass(frozen=True)
class TaskMeasurements:
    """One episode, unaggregated.

    `trials` is empty for an episode cut short because an agent could not act -
    how many turns that task needed was never established, and a 0 would read as
    a task that took no turns.
    """

    reward: float
    done: bool
    trials: int
    completion_tokens: int
    prompt_tokens: int
    intrinsic_completion_tokens: int
    intrinsic_prompt_tokens: int

    @classmethod
    def of(cls, episode: "EpisodeResult", spent: TokenTracker) -> "TaskMeasurements":
        return cls(
            reward=episode.reward,
            done=episode.done,
            trials=episode.trials,
            **_tokens(spent),
        )


# Which experiment a row belongs to. Everything here is either in the config that
# asked for the run or resolved before the first task.
IDENTITY_COLUMNS: tuple[str, ...] = ('model', 'task', 'mas_type', 'mas_memory', 'use_validator')

MEASUREMENT_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(Measurements))
TASK_MEASUREMENT_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(TaskMeasurements))

# The token counts, in the order they appear wherever they appear.
TOKEN_COLUMNS: tuple[str, ...] = tuple(_tokens(TokenTracker()))

AGGREGATE_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('max_trials',) + MEASUREMENT_COLUMNS + ('seed',)
)

TASK_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('max_trials', 'task_id') + TASK_MEASUREMENT_COLUMNS + ('seed',)
)

FAILED_TASK_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('task_id', 'error_type', 'error_message', 'seed')
)

FAILED_EXPERIMENT_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('error_type', 'error_message', 'seed')
)


TASK_RESULTS_FILENAME = '{task}-{mas_memory}-task_results.csv'
PROGRESS_FILENAME = '{task}-{mas_memory}-seed_{seed}-progress.csv'
OVERALL_RESULTS_FILENAME = 'overall_results.csv'
FAILED_TASKS_FILENAME = 'failed_tasks.csv'
FAILED_EXPERIMENTS_FILENAME = 'failed_experiments.csv'


def task_results_path(
    working_dir: str, task: str, mas_memory: str, filename: str = TASK_RESULTS_FILENAME
) -> str:
    return os.path.join(working_dir, filename.format(task=task, mas_memory=mas_memory))


def progress_path(
    working_dir: str, task: str, mas_memory: str, seed: int, filename: str = PROGRESS_FILENAME
) -> str:
    return os.path.join(
        working_dir, filename.format(task=task, mas_memory=mas_memory, seed=seed)
    )


def overall_results_path(db_dir: str, filename: str = OVERALL_RESULTS_FILENAME) -> str:
    return os.path.join(db_dir, filename)


def failed_tasks_path(working_dir: str, filename: str = FAILED_TASKS_FILENAME) -> str:
    return os.path.join(working_dir, filename)


def failed_experiments_path(db_dir: str, filename: str = FAILED_EXPERIMENTS_FILENAME) -> str:
    return os.path.join(db_dir, filename)


def identity(
    *,
    model: str,
    task: str,
    mas_type: str,
    mas_memory: str,
    use_validator: bool,
) -> dict:
    return {
        'model': model,
        'task': task,
        'mas_type': mas_type,
        'mas_memory': mas_memory,
        'use_validator': use_validator,
    }


def aggregate_row(
    *,
    identity_fields: dict,
    seed: int,
    max_trials: int,
    averages: "AggregateResults",
    tracker: TokenTracker,
) -> dict:
    """One row for the progress file or for overall_results.csv.

    `max_trials` is the budget an episode actually got, not the flag: the flag is
    None when the task's configured max_steps applies.
    """
    return {
        **identity_fields,
        'max_trials': max_trials,
        **asdict(Measurements.of(averages, tracker)),
        'seed': seed,
    }


def task_row(
    *,
    identity_fields: dict,
    seed: int,
    max_trials: int,
    task_id: int,
    episode: "EpisodeResult",
    spent: TokenTracker,
) -> dict:
    """One row for the task results file: what one episode did, unaggregated.

    `spent` is what that episode cost, not the run's running total.
    """
    return {
        **identity_fields,
        'max_trials': max_trials,
        'task_id': task_id,
        **asdict(TaskMeasurements.of(episode, spent)),
        'seed': seed,
    }


def remove_progress(path: str) -> None:
    """Drop the crash-recovery file, once the result it was protecting is written."""
    if os.path.exists(path):
        os.remove(path)


def failure_fields(error: Exception) -> dict:
    """A newline would break the row for a reader that does not parse quotes."""
    return {
        'error_type': type(error).__name__,
        'error_message': str(error).replace('\n', ' '),
    }


def write_row(path: str, columns: tuple[str, ...], row: dict, output_lock=None) -> str:
    """Append one row, writing the header first if the file is new.

    Returns the line written. Every column must be present in `row`: a row that
    does not fill its schema raises here rather than reaching the file short.
    """
    line = _format(columns, row)

    def append() -> None:
        directory = os.path.dirname(path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        new_file = not os.path.exists(path) or os.path.getsize(path) == 0
        with open(path, 'a', encoding='utf-8') as writer:
            if new_file:
                writer.write(','.join(columns) + '\n')
            writer.write(line)

    if output_lock is not None:
        with output_lock:
            append()
    else:
        append()

    return line.rstrip('\n')


def _format(columns: tuple[str, ...], row: dict) -> str:
    missing = set(columns) - set(row)
    if missing:
        raise KeyError(f'row is missing columns {sorted(missing)}')

    buffer = io.StringIO()
    csv.DictWriter(buffer, fieldnames=columns, lineterminator='\n').writerow(row)
    return buffer.getvalue()
