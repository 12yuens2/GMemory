"""The result files' schemas agree with each other.

Every file names the same identity columns first and `seed` last, and the files
reporting results share one schema, so a reader can line up a progress row, a
final row and an overall_results row column by column. Nothing in the writers
would notice the schemas drifting apart.
"""

from tasks import results

SCHEMAS = {
    'the result files': results.RESULT_COLUMNS,
    'failed_tasks.csv': results.FAILED_TASK_COLUMNS,
    'failed_experiments.csv': results.FAILED_EXPERIMENT_COLUMNS,
}


def test_every_result_schema_is_aligned():
    for name, columns in SCHEMAS.items():
        assert columns[:len(results.IDENTITY_COLUMNS)] == results.IDENTITY_COLUMNS, (
            f"{name} does not open with the identity columns"
        )
        assert columns[-1] == 'seed', f"{name} does not end in seed"
        assert len(set(columns)) == len(columns), f"{name} names a column twice"

    block = results.MEASUREMENT_COLUMNS
    start = results.RESULT_COLUMNS.index(block[0])
    assert results.RESULT_COLUMNS[start:start + len(block)] == block, (
        "the result schema does not carry the measurement block in its own order"
    )
