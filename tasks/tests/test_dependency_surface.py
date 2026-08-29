"""The third-party names the code imports are names those packages export.

conftest stubs the heavy dependencies so the package imports without a GPU, a
simulator or a network. A `MagicMock` answers to any attribute, so a stub that
permissive hides exactly the class of defect it ought to catch: `from finch import
Finch` named something `finch-clust` does not have, and the offline suite passed
anyway. Stubs whose export list is small enough to state now state it, and these
tests depend on that.
"""

import importlib
import sys


def test_the_finch_stub_carries_the_packages_real_surface():
    finch = sys.modules["finch"]

    assert hasattr(finch, "FINCH"), "finch-clust exports FINCH"
    assert not hasattr(finch, "Finch"), (
        "the stub answers to a name the package does not export, so it cannot "
        "catch a wrong import"
    )


def test_gmemory_binds_the_clustering_function_the_package_provides():
    """`from finch import Finch` is an ImportError against the pinned package."""
    gmemory = importlib.import_module("mas.memory.mas_memory.GMemory")

    assert hasattr(gmemory, "FINCH"), (
        "GMemory did not bind FINCH; check the import at the top of the module"
    )


def test_every_registered_memory_module_is_importable():
    """A module-scope import error in one memory module takes out all of them.

    `mas/memory/mas_memory/__init__.py` imports every module eagerly and
    `mas/module_map.py` reaches it through `from .memory import *`, so a bad
    import anywhere in the family stops `--mas_memory` accepting any value at all.
    """
    from mas.module_map import module_map

    try:
        module_map("io", "empty")
    except ValueError as error:
        raise AssertionError(f"the memory registry did not load: {error}") from error

    for name in ("empty", "g-memory", "chatdev", "intrinsicmemory-notemplate"):
        _, memory = module_map("io", name)
        assert memory is not None, f"{name} resolved to nothing"
