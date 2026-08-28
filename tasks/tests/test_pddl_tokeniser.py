"""The PDDL tokeniser download is scoped to PDDL (new finding, Phase 2).

pddl_env.py called nltk.download twice unconditionally: once at module scope, so
importing tasks.envs reached the network for every task and not just PDDL, and
once inside set_env, so it ran again for every task in the dataset. run.py imports
the env registry regardless of --task, so an ALFWorld sweep downloaded a PDDL
tokeniser, and an offline machine could not import the registry at all.
"""

import importlib
import sys
from unittest.mock import MagicMock

import pytest

MODULE = "tasks.envs.pddl_env.pddl_env"


@pytest.fixture
def nltk_spy(monkeypatch):
    """A fresh nltk stub, and the module reimported against it."""
    spy = MagicMock()
    spy.data.find.side_effect = LookupError("not downloaded")
    monkeypatch.setitem(sys.modules, "nltk", spy)

    module = importlib.reload(importlib.import_module(MODULE))
    module._PUNKT_READY = False
    return spy, module


def test_importing_the_module_downloads_nothing(nltk_spy):
    """The whole point: tasks.envs must be importable without a network."""
    spy, _ = nltk_spy

    spy.download.assert_not_called()


def test_the_tokeniser_is_fetched_when_a_pddl_env_is_actually_built(nltk_spy):
    spy, module = nltk_spy

    module.ensure_punkt_tokeniser()

    assert spy.download.call_count == 2, "punkt and punkt_tab are both needed"
    assert [call.args[0] for call in spy.download.call_args_list] == ["punkt", "punkt_tab"]


def test_it_is_fetched_at_most_once_per_process(nltk_spy):
    """It used to run once per task in the dataset."""
    spy, module = nltk_spy

    module.ensure_punkt_tokeniser()
    module.ensure_punkt_tokeniser()
    module.ensure_punkt_tokeniser()

    assert spy.download.call_count == 2


def test_nothing_is_downloaded_when_the_data_is_already_present(nltk_spy):
    spy, module = nltk_spy
    spy.data.find.side_effect = None

    module.ensure_punkt_tokeniser()

    spy.download.assert_not_called()


def test_set_env_asks_for_the_tokeniser(nltk_spy):
    """The one place that needs it is the one place that requests it."""
    import inspect

    _, module = nltk_spy
    source = inspect.getsource(module.PDDLEnv.set_env)

    assert "ensure_punkt_tokeniser()" in source
