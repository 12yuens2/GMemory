"""The command line offers what is registered, and defaults to nothing invalid.

Three flags select an implementation by name — `--task`, `--mas_type` and
`--mas_memory` — and each name has to exist in the corresponding registry. A name
that does not is caught late and cheaply misleading: `--mas_memory none` raised
inside `build_mas`, which `run_experiment` catches per experiment, so the default
invocation recorded a failed experiment and exited cleanly with no result rows.

Read from the registries, so a newly registered task, workflow or memory module is
covered here without editing this file.
"""

import argparse
from pathlib import Path

import pytest

from mas.module_map import MAS_MEMORY_MODULES

from tasks.envs import ENVS
from tasks.mas_workflow import MAS

# dest -> the registry its values must come from
SELECTORS = {
    "task": set(ENVS),
    "mas_type": set(MAS),
    "mas_memory": set(MAS_MEMORY_MODULES),
}

# Selecting no workflow and selecting no memory are both meaningless, so neither
# has a defensible default. `--task` keeps one.
MUST_BE_GIVEN = ["mas_type", "mas_memory"]

MINIMAL = ["--mas_type", "autogen", "--mas_memory", "empty"]


@pytest.fixture
def parser(monkeypatch) -> argparse.ArgumentParser:
    import importlib
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    return importlib.import_module("run").build_arg_parser()


def action_for(parser, dest) -> argparse.Action:
    return next(action for action in parser._actions if action.dest == dest)


def as_list(default):
    if default is None:
        return []
    return default if isinstance(default, (list, tuple)) else [default]


# ── the choices are the registry ──────────────────────────────────────────────

@pytest.mark.parametrize("dest", sorted(SELECTORS))
def test_a_selector_offers_exactly_what_is_registered(dest, parser):
    offered = action_for(parser, dest).choices

    assert offered is not None, f"--{dest} accepts any string, so a typo reaches the runner"
    assert sorted(offered) == sorted(SELECTORS[dest]), (
        f"--{dest} offers {sorted(offered)}; the registry has {sorted(SELECTORS[dest])}"
    )


@pytest.mark.parametrize("dest", sorted(SELECTORS))
def test_a_selector_never_defaults_to_something_unregistered(dest, parser):
    """`none` was the default for --mas_memory and is not a registered module.

    Registry membership only. Whether the named implementation can actually be
    constructed is a separate question - `alfworld` is registered and still cannot
    start, which is its own open finding.
    """
    for value in as_list(action_for(parser, dest).default):
        assert value in SELECTORS[dest], (
            f"--{dest} defaults to {value!r}, which is not registered"
        )


# ── the ones with no defensible default must be given ─────────────────────────

@pytest.mark.parametrize("dest", MUST_BE_GIVEN)
def test_a_selector_with_no_defensible_default_is_required(dest, parser):
    action = action_for(parser, dest)

    assert action.required, f"--{dest} is optional, so omitting it resolves to {action.default!r}"
    assert not as_list(action.default), f"--{dest} is required but still carries a default"


@pytest.mark.parametrize("dest", MUST_BE_GIVEN)
def test_omitting_a_required_selector_fails_at_parse_time(dest, parser, capsys):
    """Not inside a worker, one experiment at a time."""
    argv = [token for token in MINIMAL]
    index = argv.index(f"--{dest}")
    del argv[index:index + 2]

    with pytest.raises(SystemExit):
        parser.parse_args(argv)

    assert dest in capsys.readouterr().err


def test_the_minimal_invocation_parses(parser):
    args = parser.parse_args(MINIMAL)

    assert args.mas_type == "autogen"
    assert args.mas_memory == ["empty"]
    assert args.task == ["alfworld"]


@pytest.mark.parametrize("dest", sorted(SELECTORS))
def test_an_unregistered_value_is_rejected(dest, parser):
    argv = list(MINIMAL) + [f"--{dest}", "definitely-not-registered"]

    with pytest.raises(SystemExit):
        parser.parse_args(argv)


# ── the budget override ───────────────────────────────────────────────────────

def test_max_trials_has_no_default(parser):
    """A CLI default here silently replaced every task's configured budget."""
    assert parser.parse_args(MINIMAL).max_trials is None
