"""LLM settings, resolved on demand rather than at import.

Nothing here runs while `mas` is being imported. Importing the package therefore
needs no credentials and no particular working directory; both are resolved when
a `GPTChat` is first constructed.
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

from .utils import load_config

# Resolved against this file, not the working directory, so the process can be
# started from any directory.
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

        An exported environment variable takes precedence over .env.
        """
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
    """The settings used by any GPTChat not given its own.

    Cached: a sweep builds a GPTChat per experiment and per memory reset, and
    none of them need to re-read the config file.
    """
    global _CACHED
    if _CACHED is None:
        _CACHED = LLMSettings.load()
    return _CACHED


def reset_default_llm_settings() -> None:
    """Drop the cache. For tests that change the environment between cases."""
    global _CACHED
    _CACHED = None
