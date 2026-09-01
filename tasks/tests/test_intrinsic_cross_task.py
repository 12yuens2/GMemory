"""Whether an intrinsic memory outlives the task that built it.

The baseline arms are within-episode only: every task starts from nothing, which
is the contrast with the memory modules that accumulate across a dataset. This
is the second arm - the same mechanism with a longer life - so the property that
matters is that the two are distinguishable, both in the memory and in a result
row.
"""

import tempfile

import pytest

from mas.llm import TokenTracker
from mas.module_map import MAS_MEMORY_MODULES

from tasks.tests.fakes import FakeEmbeddingFunc, FakeLLM
from tasks.tests.test_intrinsic_tokens import INTRINSIC_KEYS, TRAJECTORY


def build_with_memory(key: str, **global_config):
    """An intrinsic memory that has already summarised one task's trajectory."""
    memory = MAS_MEMORY_MODULES[key](
        namespace=key,
        global_config={'working_dir': tempfile.mkdtemp(), 'hop': 1, **global_config},
        llm_model=FakeLLM(replies=['what the agent learned'], tracker=TokenTracker()),
        embedding_func=FakeEmbeddingFunc(),
    )
    memory.init_task_context('put a mug in the cabinet', task_description='a description')
    for action, observation in TRAJECTORY:
        memory.move_memory_state(action, observation)
    memory.summarize(solver_message='the solver said this')

    assert memory.agent_intrinsic_memory, f'{key} did not build a memory to carry'
    return memory


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_a_finished_task_leaves_nothing_behind_by_default(key):
    memory = build_with_memory(key)

    memory.save_task_context(label=True)

    assert memory.agent_intrinsic_memory == '', (
        f'{key} carried memory into the next task without being asked to'
    )


@pytest.mark.parametrize('key', INTRINSIC_KEYS)
def test_the_memory_survives_a_finished_task_when_asked_to(key):
    memory = build_with_memory(key, intrinsic_cross_task=True)

    memory.save_task_context(label=True)

    assert memory.agent_intrinsic_memory == 'what the agent learned', (
        f'{key} was asked to carry its memory across tasks and wiped it anyway'
    )


def test_the_setting_names_itself_in_every_result_row():
    """Without a column, the two arms are rows with the same mas_memory value."""
    import sys
    from pathlib import Path

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    import results

    assert 'intrinsic_cross_task' in results.IDENTITY_COLUMNS
