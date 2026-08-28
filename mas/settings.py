"""Explicit settings, resolved on demand rather than at import.

`mas/llm.py` used to read the environment and configs/configs.yaml while being
imported:

    CONFIG = load_config("configs/configs.yaml")
    URL = os.environ["OPENAI_API_BASE"]
    KEY = os.environ["OPENAI_API_KEY"]

Three consequences. The config path was relative to the working directory, so the
process could only be started from the repository root. `os.environ[...]` raised
KeyError with no indication of what to set. And `mas/__init__.py` had to run
first, because it was what loaded .env - so importing `mas.llm` directly, or
importing in the wrong order, failed. tasks/tests/conftest.py existed to work
around exactly that.

Reading settings when they are first needed removes all three. Importing mas no
longer needs credentials at all, which is also what lets the offline suite import
the whole package.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .utils import load_config

# Both resolved against this file, not the working directory, so the process can
# be started from anywhere.
_REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = _REPO_ROOT / "configs" / "configs.yaml"
DEFAULT_ENV_PATH = _REPO_ROOT / ".env"


class MissingSettings(RuntimeError):
    """A required setting is absent, named so the reader knows what to set."""


@dataclass(frozen=True)
class LLMSettings:
    """Everything GPTChat needs to reach an endpoint and size a request."""

    api_base: str
    api_key: str
    max_tokens: int = 512
    temperature: float = 0.1
    num_comps: int = 1

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "LLMSettings":
        """Build settings from .env, the environment and configs/configs.yaml.

        Environment wins over .env, which is what load_dotenv does by default:
        an exported variable should beat a checked-out file.
        """
        # Named explicitly, rather than letting load_dotenv walk up from the
        # working directory, so a run started elsewhere still finds it.
        load_dotenv(dotenv_path=DEFAULT_ENV_PATH)

        api_base = os.getenv("OPENAI_API_BASE")
        api_key = os.getenv("OPENAI_API_KEY")
        missing = [
            name
            for name, value in (("OPENAI_API_BASE", api_base), ("OPENAI_API_KEY", api_key))
            if not value
        ]
        if missing:
            raise MissingSettings(
                f"{' and '.join(missing)} must be set, in the environment or in a .env "
                f"file at the repository root."
            )

        llm_config: dict = (load_config(str(config_path or DEFAULT_CONFIG_PATH)) or {}).get(
            "llm_config", {}
        )
        return cls(
            api_base=api_base,
            api_key=api_key,
            max_tokens=llm_config.get("max_token", 512),
            temperature=llm_config.get("temperature", 0.1),
            num_comps=llm_config.get("num_comps", 1),
        )


_CACHED: Optional[LLMSettings] = None


def default_llm_settings() -> LLMSettings:
    """The settings used by any GPTChat not given its own, loaded once per process.

    Cached because a sweep builds one GPTChat per experiment and per memory reset,
    and none of them should re-read the config file.
    """
    global _CACHED
    if _CACHED is None:
        _CACHED = LLMSettings.load()
    return _CACHED


def reset_default_llm_settings() -> None:
    """Drop the cache. For tests that change the environment between cases."""
    global _CACHED
    _CACHED = None
