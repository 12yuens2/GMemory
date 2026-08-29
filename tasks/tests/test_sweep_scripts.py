"""The Slurm sweep scripts request configurations the CLI recognises, once each.

A sweep script is the only place an experiment grid is written down, and a mistake
in one is invisible in the results: a module listed twice produces two rows per
configuration in overall_results.csv, which a group-by averages together, and a
name the registry does not know fails only once the job is already queued.

Read from the scripts and the registries, so a new module or task is covered as
soon as it is registered.
"""

import re
from pathlib import Path

import pytest

from mas.module_map import module_map

from tasks.envs import ENVS

SLURM = Path(__file__).resolve().parents[2] / "slurm"
SCRIPTS = sorted(path.name for path in SLURM.glob("*.sh"))


def flag_values(script: str, flag: str) -> list[str]:
    """The values given to `flag`, across a line-continued shell invocation.

    Values run until the next flag, so this reads a variadic argument the way the
    shell hands it to argparse.
    """
    joined = re.sub(r"\\\s*\n", " ", (SLURM / script).read_text())
    match = re.search(rf"{re.escape(flag)}\s+(.*)", joined)
    if match is None:
        return []

    values = []
    for token in match.group(1).split():
        if token.startswith("-"):
            break
        values.append(token)
    return values


def registered_memory_modules() -> set[str]:
    """The names module_map accepts, read out of the error it raises for a bad one."""
    try:
        module_map("io", "definitely-not-a-memory-module")
    except ValueError as error:
        return set(re.findall(r"'([\w-]+)'", str(error).split("Allowed values:")[-1]))
    raise AssertionError("module_map accepted an unregistered memory module")


@pytest.mark.parametrize("script", SCRIPTS)
def test_no_memory_module_is_requested_twice(script):
    requested = flag_values(script, "--mas_memory")
    duplicates = {name for name in requested if requested.count(name) > 1}

    assert not duplicates, (
        f"{script} requests {sorted(duplicates)} more than once; at "
        f"{len(flag_values(script, '--seed')) or 1} seeds that is "
        f"{len(duplicates) * (len(flag_values(script, '--seed')) or 1)} redundant "
        f"experiments and a duplicate row per configuration"
    )


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_requested_memory_module_is_registered(script):
    registered = registered_memory_modules() | {"none"}
    requested = flag_values(script, "--mas_memory")

    unknown = [name for name in requested if name not in registered]
    assert not unknown, f"{script} requests {unknown}, which module_map does not accept"


@pytest.mark.parametrize("script", SCRIPTS)
def test_every_requested_task_is_registered(script):
    requested = flag_values(script, "--task")

    unknown = [name for name in requested if name not in ENVS]
    assert not unknown, f"{script} requests {unknown}, which is not a registered task"


@pytest.mark.parametrize("script", SCRIPTS)
def test_no_seed_is_requested_twice(script):
    requested = flag_values(script, "--seed")
    duplicates = {seed for seed in requested if requested.count(seed) > 1}

    assert not duplicates, f"{script} repeats seed(s) {sorted(duplicates)}"
