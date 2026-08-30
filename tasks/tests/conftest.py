"""
conftest.py — executed before test collection.

Stubs the heavy or unavailable dependencies that mas and tasks.envs import at
module scope, so the whole package can be imported offline.

Also tees all stdout/stderr output to a timestamped .txt file in tasks/tests/logs/
so that a full record of every run (including -s print output) is preserved.
"""

import sys
import os
from pathlib import Path
from datetime import datetime
from types import ModuleType
from unittest.mock import MagicMock

# An endpoint no test will reach, so a developer's .env cannot leak into a run.
os.environ["OPENAI_API_BASE"] = "http://localhost:9999"
os.environ["OPENAI_API_KEY"] = "test-key"


def _stub_module(name: str):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


def _stub_module_exporting(name: str, *exports: str):
    """Stub a module exposing only `exports`, so a wrong import name still fails."""
    if name in sys.modules:
        return
    module = ModuleType(name)
    for export in exports:
        setattr(module, export, MagicMock(name=f"{name}.{export}"))
    module.__all__ = list(exports)
    sys.modules[name] = module


# The task environments import their simulators at module scope, so the env and
# recorder registries need these stubbed to be importable offline.
for _mod in (
    "langchain_chroma",
    "langchain_core",
    "langchain_core.documents",
    "nltk",
    "pddlgym",
    "pddlgym.structs",
    "scienceworld",
    "wikipedia",
):
    _stub_module(_mod)

# finch-clust's whole public surface is one function.
_stub_module_exporting("finch", "FINCH")


# ── output logging ────────────────────────────────────────────────────────────

class _TeeWriter:
    """Writes to multiple streams simultaneously."""

    def __init__(self, *streams):
        self._streams = streams

    def write(self, text):
        for s in self._streams:
            s.write(text)

    def flush(self):
        for s in self._streams:
            s.flush()

    def fileno(self):
        return self._streams[0].fileno()

    def isatty(self):
        return getattr(self._streams[0], "isatty", lambda: False)()

    def __getattr__(self, name):
        return getattr(self._streams[0], name)


def pytest_configure(config):
    log_dir = Path(__file__).parent / "logs"
    log_dir.mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"{timestamp}.txt"

    log_file = log_path.open("w", encoding="utf-8")
    config._tee_log_file = log_file
    config._tee_log_path = log_path

    config._orig_stdout = sys.stdout
    config._orig_stderr = sys.stderr
    sys.stdout = _TeeWriter(sys.stdout, log_file)
    sys.stderr = _TeeWriter(sys.stderr, log_file)


def pytest_unconfigure(config):
    if hasattr(config, "_orig_stdout"):
        sys.stdout = config._orig_stdout
    if hasattr(config, "_orig_stderr"):
        sys.stderr = config._orig_stderr
    if hasattr(config, "_tee_log_file"):
        config._tee_log_file.close()
        print(f"\nTest output saved to: {config._tee_log_path}")
