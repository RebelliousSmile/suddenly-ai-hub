"""Provider factory — creates the right provider based on env vars.

Priority order:
  1. Explicit provider_name argument
  2. SUPPORTING_PROVIDER env var
  3. First available provider with a valid API key (auto-detect)

Environment variables:
  SUPPORTING_PROVIDER    — "together" or "fireworks" (optional)
  TOGETHER_API_KEY       — Together.ai API key
  FIREWORKS_API_KEY      — Fireworks.ai API key

Usage:
  from evaluation.providers import get_provider
  provider = get_provider("together")       # explicit
  provider = get_provider()                  # env var or auto-detect
"""

from __future__ import annotations

import os
from typing import Optional

from .base import BaseProvider
from .fireworks import FireworksProvider
from .together import TogetherProvider

PROVIDER_ENV_VAR = "SUPPORTING_PROVIDER"
TOGETHER_API_KEY_ENV = "TOGETHER_API_KEY"
FIREWORKS_API_KEY_ENV = "FIREWORKS_API_KEY"

PROVIDERS: dict[str, type[BaseProvider]] = {
    "together": TogetherProvider,
    "fireworks": FireworksProvider,
}

API_KEY_ENV_MAP: dict[str, str] = {
    "together": TOGETHER_API_KEY_ENV,
    "fireworks": FIREWORKS_API_KEY_ENV,
}


def _is_key_valid(provider_name: str, key: str) -> bool:
    """Check API key validity without instantiating the provider.

    This avoids creating objects just to validate, and prevents
    accidental side-effects from __init__.
    """
    if not key:
        return False
    if provider_name == "together":
        # Keys may be "sk-..." (old) or "tgp_v1-..." (new)
        return len(key) > 10
    if provider_name == "fireworks":
        return len(key) > 5
    return False


def get_provider(provider_name: Optional[str] = None) -> BaseProvider:
    """Create and return a configured provider instance.

    Args:
        provider_name: Explicit provider name ("together" or "fireworks").
                       If None, falls back to SUPPORTING_PROVIDER env var,
                       then to auto-detection.

    Returns:
        A configured BaseProvider instance.

    Raises:
        ValueError: If the provider is unknown, no API key is found,
                    or the key is invalid.
    """
    # 1. Explicit argument
    if provider_name:
        provider_name = provider_name.lower()
        if provider_name not in PROVIDERS:
            raise ValueError(
                f"Unknown provider '{provider_name}'. "
                f"Available: {list(PROVIDERS.keys())}"
            )
        env_var = API_KEY_ENV_MAP[provider_name]
        api_key = os.getenv(env_var, "")
        if not _is_key_valid(provider_name, api_key):
            raise ValueError(
                f"No valid API key found for provider '{provider_name}'. "
                f"Set {env_var}."
            )
        return PROVIDERS[provider_name](api_key=api_key)

    # 2. SUPPORTING_PROVIDER env var
    env_provider = os.getenv(PROVIDER_ENV_VAR, "").lower()
    if env_provider in PROVIDERS:
        return get_provider(env_provider)

    # 3. Auto-detect: first provider with a valid key
    for name in PROVIDERS:
        env_var = API_KEY_ENV_MAP[name]
        api_key = os.getenv(env_var, "")
        if _is_key_valid(name, api_key):
            return PROVIDERS[name](api_key=api_key)

    raise ValueError(
        "No provider available. Set SUPPORTING_PROVIDER=together or fireworks, "
        "or ensure TOGETHER_API_KEY / FIREWORKS_API_KEY are set."
    )


def list_available_providers() -> list[str]:
    """Return list of available provider names."""
    return list(PROVIDERS.keys())
