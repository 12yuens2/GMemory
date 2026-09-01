"""The result rows an experiment writes, and the CSV files they go in.

Three rules hold across every file here: the same identity columns come first,
`seed` is last, and the two files reporting results report the same columns in
the same order. A reader can then line up a progress row, a final row and a row
of `overall_results.csv` column by column, and nothing has to recover a field by
its position.

The progress file is a partial result kept on purpose. One row is appended per
completed task, holding the means over the tasks scored *so far*, so a job the
scheduler kills part-way through a dataset still has what it measured up to
there. `tasks_scored` is that row's denominator; the last row of a run that
finished is the run's result, and matches the row `results.csv` gets.
"""

import csv
import io
import os
from dataclasses import asdict, dataclass, fields
from typing import TYPE_CHECKING

from mas.llm import TokenTracker

if TYPE_CHECKING:
    from tasks.envs.base_env import AggregateResults


@dataclass(frozen=True)
class Measurements:
    """What episodes measured, shared by every file that reports results.

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
            completion_tokens=tracker.completion_tokens,
            prompt_tokens=tracker.prompt_tokens,
            intrinsic_completion_tokens=tracker.intrinsic_completion_tokens,
            intrinsic_prompt_tokens=tracker.intrinsic_prompt_tokens,
        )


# Which experiment a row belongs to. Everything here is either in the config that
# asked for the run or resolved before the first task.
IDENTITY_COLUMNS: tuple[str, ...] = ('model', 'task', 'mas_type', 'mas_memory', 'use_validator')

MEASUREMENT_COLUMNS: tuple[str, ...] = tuple(field.name for field in fields(Measurements))

RESULT_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('max_trials',) + MEASUREMENT_COLUMNS + ('seed',)
)

FAILED_TASK_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('task_id', 'error_type', 'error_message', 'seed')
)

FAILED_EXPERIMENT_COLUMNS: tuple[str, ...] = (
    IDENTITY_COLUMNS + ('error_type', 'error_message', 'seed')
)


def progress_path(working_dir: str, task: str, mas_memory: str) -> str:
    return os.path.join(working_dir, f'{task}-{mas_memory}-progress.csv')


def results_path(working_dir: str) -> str:
    return os.path.join(working_dir, 'results.csv')


def overall_results_path(db_dir: str) -> str:
    return os.path.join(db_dir, 'overall_results.csv')


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


def result_row(
    *,
    identity_fields: dict,
    seed: int,
    max_trials: int,
    averages: "AggregateResults",
    tracker: TokenTracker,
) -> dict:
    """One row for the progress file, results.csv or overall_results.csv.

    `max_trials` is the budget an episode actually got, not the flag: the flag is
    None when the task's configured max_steps applies.
    """
    return {
        **identity_fields,
        'max_trials': max_trials,
        **asdict(Measurements.of(averages, tracker)),
        'seed': seed,
    }


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
