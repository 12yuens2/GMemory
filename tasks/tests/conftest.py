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
from unittest.mock import MagicMock

# Importing mas no longer needs credentials - mas.settings reads them when a
# GPTChat is first constructed. These are here only so a test that does construct
# one gets an endpoint it will never reach rather than a real one from a
# developer's .env.
os.environ["OPENAI_API_BASE"] = "http://localhost:9999"
os.environ["OPENAI_API_KEY"] = "test-key"


def _stub_module(name: str):
    if name not in sys.modules:
        sys.modules[name] = MagicMock()


# The task environments import their simulators at module scope, so importing the
# env/recorder registry offline needs these stubbed. nltk in particular downloads a
# tokeniser at import time, which a test run must not depend on.
for _mod in (
    "langchain_chroma",
    "langchain_core",
    "langchain_core.documents",
    "finch",
    "nltk",
    "pddlgym",
    "pddlgym.structs",
    "scienceworld",
    "wikipedia",
):
    _stub_module(_mod)


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
