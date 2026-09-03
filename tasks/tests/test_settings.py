"""Settings are resolved on demand, not at import, and come from the run's flags.

Importing mas must need no credentials and no particular working directory, a
missing variable must say which one it is, and nothing may invent a token
budget or a temperature of its own: the run installs them, and a GPTChat built
before that must say so rather than pick a number.
"""

import subprocess
import sys
from pathlib import Path

import pytest

from mas.settings import (
    DEFAULT_ENV_PATH,
    LLMSettings,
    MissingSettings,
    default_llm_settings,
    reset_default_llm_settings,
    use_llm_settings,
)

REPO_ROOT = Path(__file__).resolve().parents[2]

CALL_SETTINGS = dict(max_tokens=512, temperature=0.1, request_timeout=300.0, log_responses=False)


@pytest.fixture(autouse=True)
def clean_settings_cache():
    reset_default_llm_settings()
    yield
    reset_default_llm_settings()


def test_the_env_path_is_absolute():
    assert DEFAULT_ENV_PATH.is_absolute()


def test_missing_credentials_name_themselves(monkeypatch, tmp_path):
    monkeypatch.delenv("OPENAI_API_BASE", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("mas.settings.DEFAULT_ENV_PATH", tmp_path / "absent.env")

    with pytest.raises(MissingSettings, match="OPENAI_API_BASE and OPENAI_API_KEY"):
        LLMSettings.load(**CALL_SETTINGS)


def test_only_the_absent_variable_is_named(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:9999")
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr("mas.settings.DEFAULT_ENV_PATH", tmp_path / "absent.env")

    with pytest.raises(MissingSettings, match="^OPENAI_API_KEY must be set"):
        LLMSettings.load(**CALL_SETTINGS)


def test_a_base_url_with_no_scheme_is_refused(monkeypatch, tmp_path):
    """httpx parses a bare host:port as a relative path, with no host at all."""
    monkeypatch.setenv("OPENAI_API_BASE", "10.0.0.1:8000")
    monkeypatch.setenv("OPENAI_API_KEY", "none")
    monkeypatch.setattr("mas.settings.DEFAULT_ENV_PATH", tmp_path / "absent.env")

    with pytest.raises(MissingSettings, match="http://"):
        LLMSettings.load(**CALL_SETTINGS)


def test_the_call_settings_are_the_ones_the_caller_asked_for(monkeypatch):
    monkeypatch.setenv("OPENAI_API_BASE", "http://localhost:9999")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    settings = LLMSettings.load(
        max_tokens=128, temperature=0.7, request_timeout=30.0, log_responses=True
    )

    assert (settings.max_tokens, settings.temperature) == (128, 0.7)
    assert (settings.request_timeout, settings.log_responses) == (30.0, True)


def test_no_call_setting_has_a_default_of_its_own():
    """Every one of them is the run's to choose, so none may be omitted."""
    for omitted in CALL_SETTINGS:
        incomplete = {name: value for name, value in CALL_SETTINGS.items() if name != omitted}
        with pytest.raises(TypeError, match=omitted):
            LLMSettings(api_base="http://localhost:9999", api_key="none", **incomplete)


def test_a_gptchat_uses_the_settings_the_run_installed():
    """The route from a flag to a request: nothing else passes settings down."""
    from mas.llm import GPTChat

    use_llm_settings(
        LLMSettings(
            api_base="http://localhost:9999/v1",
            api_key="none",
            max_tokens=77,
            temperature=0.5,
            request_timeout=42.0,
            log_responses=False,
        )
    )

    assert GPTChat(model_name="fake-model").settings.max_tokens == 77
    assert default_llm_settings().temperature == 0.5


def test_asking_for_settings_before_a_run_installs_them_is_an_error():
    """Answering with a token budget of its own is how a flag goes silently unused."""
    with pytest.raises(MissingSettings, match="install_llm_settings"):
        default_llm_settings()


def test_importing_mas_llm_needs_no_credentials_and_no_working_directory():
    """A subprocess with the variables stripped, started outside the repo."""
    result = subprocess.run(
        [sys.executable, "-c", f"import sys; sys.path.insert(0, {str(REPO_ROOT)!r}); import mas.llm"],
        cwd="/",
        env={"PATH": "/usr/bin:/bin", "HOME": "/tmp"},
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, f"importing mas.llm failed:\n{result.stderr}"
