"""The third-party names the code imports are names those packages export.

conftest stubs the heavy dependencies, and a `MagicMock` answers to any
attribute - so stubs whose export list is small enough to state state it, and
these tests depend on that.
"""

import importlib
import sys

import pytest

from mas.module_map import MAS_MEMORY_MODULES


def test_the_finch_stub_carries_the_packages_real_surface():
    finch = sys.modules["finch"]

    assert hasattr(finch, "FINCH"), "finch-clust exports FINCH"
    assert not hasattr(finch, "Finch"), (
        "the stub answers to a name the package does not export, so it cannot "
        "catch a wrong import"
    )


def test_gmemory_binds_the_clustering_function_the_package_provides():
    gmemory = importlib.import_module("mas.memory.mas_memory.GMemory")

    assert hasattr(gmemory, "FINCH"), (
        "GMemory did not bind FINCH; check the import at the top of the module"
    )


@pytest.mark.parametrize("memory_key", sorted(MAS_MEMORY_MODULES))
def test_every_registered_memory_module_resolves(memory_key):
    """A module-scope import error in one memory module takes out all of them.

    The family is imported eagerly, so a bad import anywhere in it stops
    `--mas_memory` accepting any value.
    """
    from mas.module_map import module_map

    _, memory = module_map("io", memory_key)

    assert memory is not None, f"{memory_key} resolved to nothing"
    assert isinstance(memory, type), f"{memory_key} resolved to {memory!r}, not a class"
