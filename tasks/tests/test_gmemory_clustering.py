"""Every task node in g-memory's graph comes out of clustering with a cluster id.

`merge_insights` runs every twentieth task and reaches `cluster_tasks`, so a
failure here is not a degraded clustering - it raises out of `save_task_context`,
out of `schedule`, and the task is recorded as failed and unscored after its
episode has run every trial and spent every token.

FINCH's third return value is None unless `req_clust` is passed, which is what
made that happen. It is faked here rather than run, so the test pins how the
return value is read, which is the part that was wrong.
"""

import importlib
from types import SimpleNamespace

import numpy as np
import pytest

# The package __init__ re-exports the GMemory class under the module's own name.
gmemory_module = importlib.import_module('mas.memory.mas_memory.GMemory')

TASKS = ['boil water', 'freeze water', 'find the longest-lived animal']

# What FINCH really returns: one column per partition, finest first, and None for
# the third value unless a caller asks for a specific number of clusters.
PARTITIONS = np.array([[0, 0], [1, 0], [2, 1]])
FINCH_RETURN = (PARTITIONS, [3, 2], None)


def build_task_layer(tmp_path, nodes=TASKS):
    layer = gmemory_module.TaskLayer(
        working_dir=str(tmp_path),
        namespace='g-memory',
        task_storage=SimpleNamespace(
            _embedding_function=SimpleNamespace(embed_query=lambda text: [0.1, 0.2, 0.3])
        ),
    )
    for node in nodes:
        layer.graph.add_node(node)
    return layer


@pytest.fixture
def finch(monkeypatch):
    calls = []

    def fake_finch(data, **kwargs):
        calls.append((data, kwargs))
        return FINCH_RETURN

    monkeypatch.setattr(gmemory_module, 'FINCH', fake_finch)
    return calls


def test_every_node_is_given_an_integer_cluster_id(finch, tmp_path):
    layer = build_task_layer(tmp_path)

    layer.cluster_tasks()

    ids = [layer.graph.nodes[node].get('cluster_id') for node in layer.graph.nodes]
    assert ids == [0, 1, 2], f"the finest partition assigns one id per node, got {ids}"


def test_the_ids_are_the_ones_merge_insights_iterates(finch, tmp_path):
    """merge_insights raises RuntimeError on a None label, so this is its contract."""
    layer = build_task_layer(tmp_path)

    layer.cluster_tasks()

    assert [label for _, label in layer] == [0, 1, 2]


def test_a_clustering_that_fails_still_leaves_every_node_labelled(monkeypatch, tmp_path):
    def explode(data, **kwargs):
        raise ValueError('not enough samples')

    monkeypatch.setattr(gmemory_module, 'FINCH', explode)
    layer = build_task_layer(tmp_path)

    layer.cluster_tasks()

    assert [label for _, label in layer] == [0, 0, 0], (
        'the fallback puts everything in one cluster, which is a degraded '
        'clustering rather than a failed task'
    )
