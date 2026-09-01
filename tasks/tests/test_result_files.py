"""One results file, written by more processes than share anything.

An experiment set is submitted as several Slurm jobs, all pointed at one
--db_dir, so overall_results.csv is appended to by processes in different jobs.
They share no process pool, so the header check and the append have to be one
operation against the file itself.

The two ways this goes wrong are both unrecoverable by the reader: a second
header in the middle of the file, and rows of one schema under another schema's
header.
"""

import fcntl
import multiprocessing
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import results  # noqa: E402

COLUMNS = ('model', 'task', 'seed')
ROWS_PER_WRITER = 30


def write_rows(path: str, writer_id: int) -> None:
    """One process's share of the rows. Module level, so spawn can pickle it."""
    for row_id in range(ROWS_PER_WRITER):
        results.write_row(
            path, COLUMNS, {'model': f'writer-{writer_id}', 'task': 'fever', 'seed': row_id}
        )


def test_nothing_is_appended_while_another_job_holds_the_file(tmp_path):
    """The lock has to be on the file: two jobs share no process pool to lock in.

    The parent takes the lock a job in another allocation would hold, and the
    writer must wait rather than deciding for itself whether the file is empty.
    """
    path = tmp_path / 'overall_results.csv'
    path.write_text(','.join(COLUMNS) + '\n')
    ctx = multiprocessing.get_context('spawn')
    writer = ctx.Process(target=write_rows, args=(str(path), 0))

    with path.open('a') as holder:
        fcntl.flock(holder.fileno(), fcntl.LOCK_EX)
        writer.start()
        writer.join(timeout=3)
        written_while_locked = path.read_text().splitlines()
        fcntl.flock(holder.fileno(), fcntl.LOCK_UN)

    writer.join(timeout=30)

    assert written_while_locked == [','.join(COLUMNS)], (
        'a row was appended while another job held the lock'
    )
    lines = path.read_text().splitlines()
    assert lines.count(','.join(COLUMNS)) == 1, 'a header was written into the middle of the file'
    assert len(lines) == 1 + ROWS_PER_WRITER, 'a row was lost once the lock was released'


def test_a_file_written_by_an_earlier_schema_is_refused(tmp_path):
    path = tmp_path / 'overall_results.csv'
    path.write_text('model,task\nfake-model,fever\n')

    with pytest.raises(results.SchemaMismatch, match='overall_results.csv'):
        results.write_row(str(path), COLUMNS, {'model': 'm', 'task': 'fever', 'seed': 1})


def test_a_file_with_the_right_header_is_appended_to(tmp_path):
    path = tmp_path / 'overall_results.csv'
    results.write_row(str(path), COLUMNS, {'model': 'm', 'task': 'fever', 'seed': 1})
    results.write_row(str(path), COLUMNS, {'model': 'm', 'task': 'fever', 'seed': 2})

    lines = path.read_text().splitlines()
    assert len(lines) == 3
    assert lines.count(','.join(COLUMNS)) == 1


def test_an_existing_file_from_another_schema_is_refused_before_the_sweep_starts(tmp_path):
    """Otherwise it is found hours in, when the first experiment finishes."""
    path = tmp_path / 'overall_results.csv'

    results.check_header(str(path), COLUMNS)  # absent: nothing to disagree with

    path.write_text('model,task\n')
    with pytest.raises(results.SchemaMismatch, match='overall_results.csv'):
        results.check_header(str(path), COLUMNS)
