"""Settings are resolved on demand, not at import (test plan item, Phase 2 step 6).

mas/llm.py used to read os.environ["OPENAI_API_BASE"] and
load_config("configs/configs.yaml") while being imported. So importing mas
required credentials, the config path was relative to the working directory, and
import order was load-bearing - mas/__init__.py had to run first because it was
what loaded .env. tasks/tests/conftest.py existed to work around it.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from mas.settings import (
    DEFAULT_CONFIG_PATH,
    DEFAULT_ENV_PATH,
    LLMSettings,
    MissingSettings,
    default_llm_settings,
    reset_default_llm_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(autouse=True)
def clean_settings_cache():
    reset_default_llm_settings()
    yield
    reset_default_llm_settings()


def test_the_config_path_is_absolute_and_exists():
    """It was "configs/configs.yaml", which bound the process to one directory."""
    assert DEFAULT_CONFIG_PATH.is_absolute()
    assert DEFAULT_CONFIG_PATH.exists()
    assert DEFAULT_ENV_PATH.is_absolute()


def test_missing_credentials_name_themselves(monkeypatch, tmp_path):
    """os.environ[...] raised a bare KeyError, and mas/__init__.py's
    os.environ[x] = os.getenv(x) raised TypeError - neither said what to set."""
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("mas.settings.DEFAULT_ENV_PATH", tmp_path / "absent.env")

    with pytest.raises(MissingSettings, match="OPENAI_API_BASE and OPENAI_API_KEY"):
        LLMSettings.load()


def test_only_the_absent_variable_is_named(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:9999")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("mas.settings.DEFAULT_ENV_PATH", tmp_path / "absent.env")

    with pytest.raises(MissingSettings, match="^OPENAI_API_KEY must be set"):
        LLMSettings.load()


def test_the_llm_config_is_read_from_the_yaml(monkeypatch, tmp_path):
    config = tmp_path / "configs.yaml"
    config.write_text("llm_config:\n  max_token: 128\n  temperature: 0.7\n  num_comps: 4\n")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:9999")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = LLMSettings.load(config_path=config)

    assert (settings.max_tokens, settings.temperature, settings.num_comps) == (128, 0.7, 4)


def test_the_defaults_survive_a_config_with_no_llm_block(monkeypatch, tmp_path):
    config = tmp_path / "configs.yaml"
    config.write_text("something_else: 1\n")
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:9999")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = LLMSettings.load(config_path=config)

    assert (settings.max_tokens, settings.temperature, settings.num_comps) == (512, 0.1, 1)


def test_the_default_settings_are_loaded_once_per_process():
    """A sweep builds a GPTChat per experiment and per memory reset; none of them
    should re-read the config file."""
    assert default_llm_settings() is default_llm_settings()


def test_importing_mas_llm_needs_no_credentials_and_no_working_directory():
    """The real regression guard: a subprocess with the variables stripped,
    started somewhere other than the repository root."""
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); import mas.llm"],
        cwd="/",
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"importing mas.llm failed:\n{result.stderr}"
