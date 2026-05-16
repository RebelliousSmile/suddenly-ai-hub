"""Tests for provider factory."""

import os
import pytest
from pipelines.evaluation.providers.factory import (
    _is_key_valid,
    get_provider,
    list_available_providers,
    PROVIDER_ENV_VAR,
)


class TestListAvailableProviders:
    def test_returns_both_providers(self):
        available = list_available_providers()
        assert "together" in available
        assert "fireworks" in available
        assert len(available) == 2

    def test_returns_list_not_dict(self):
        available = list_available_providers()
        assert isinstance(available, list)


class TestIsKeyValid:
    def test_together_valid_key(self):
        assert _is_key_valid("together", "sk-test-key-123") is True

    def test_together_empty_key(self):
        assert _is_key_valid("together", "") is False

    def test_together_short_key(self):
        assert _is_key_valid("together", "ab") is False

    def test_fireworks_valid_key(self):
        assert _is_key_valid("fireworks", "fw-key-123") is True

    def test_fireworks_empty_key(self):
        assert _is_key_valid("fireworks", "") is False

    def test_fireworks_short_key(self):
        assert _is_key_valid("fireworks", "abc") is False

    def test_fireworks_long_key(self):
        assert _is_key_valid("fireworks", "a" * 20) is True

    def test_unknown_provider(self):
        assert _is_key_valid("unknown", "any-key") is False


class TestGetProviderExplicit:
    def test_together_provider_explicit(self):
        os.environ["TOGETHER_API_KEY"] = "sk-test-key-123"
        try:
            p = get_provider("together")
            assert p.name == "together"
            assert p.validate() is True
        finally:
            del os.environ["TOGETHER_API_KEY"]

    def test_fireworks_provider_explicit(self):
        os.environ["FIREWORKS_API_KEY"] = "fw-key-123"
        try:
            p = get_provider("fireworks")
            assert p.name == "fireworks"
            assert p.validate() is True
        finally:
            del os.environ["FIREWORKS_API_KEY"]

    def test_unknown_provider_raises(self):
        with pytest.raises(ValueError, match="Unknown provider"):
            get_provider("unknown")

    def test_case_insensitive_provider_name(self):
        os.environ["TOGETHER_API_KEY"] = "sk-test-key-123"
        try:
            p = get_provider("Together")
            assert p.name == "together"
        finally:
            del os.environ["TOGETHER_API_KEY"]


class TestGetProviderEnvVar:
    def test_reads_supporting_provider_together(self):
        os.environ["TOGETHER_API_KEY"] = "sk-test-key-123"
        os.environ["SUPPORTING_PROVIDER"] = "together"
        try:
            p = get_provider()
            assert p.name == "together"
        finally:
            del os.environ["SUPPORTING_PROVIDER"]
            del os.environ["TOGETHER_API_KEY"]

    def test_reads_supporting_provider_fireworks(self):
        os.environ["FIREWORKS_API_KEY"] = "fw-key-123"
        os.environ["SUPPORTING_PROVIDER"] = "fireworks"
        try:
            p = get_provider()
            assert p.name == "fireworks"
        finally:
            del os.environ["SUPPORTING_PROVIDER"]
            del os.environ["FIREWORKS_API_KEY"]

    def test_env_var_takes_precedence_over_argument(self):
        """When SUPPORTING_PROVIDER is set and no arg given, use env."""
        os.environ["TOGETHER_API_KEY"] = "sk-test-key-123"
        os.environ["SUPPORTING_PROVIDER"] = "together"
        try:
            p = get_provider()  # no arg
            assert p.name == "together"
        finally:
            del os.environ["SUPPORTING_PROVIDER"]
            del os.environ["TOGETHER_API_KEY"]

    def test_explicit_arg_overrides_env(self):
        """Explicit arg should take precedence over env var."""
        os.environ["TOGETHER_API_KEY"] = "sk-test-key-123"
        os.environ["FIREWORKS_API_KEY"] = "fw-key-123"
        os.environ["SUPPORTING_PROVIDER"] = "together"
        try:
            p = get_provider("fireworks")  # explicit fireworks
            assert p.name == "fireworks"
        finally:
            del os.environ["SUPPORTING_PROVIDER"]
            del os.environ["TOGETHER_API_KEY"]
            del os.environ["FIREWORKS_API_KEY"]

    def test_no_valid_key_raises(self):
        os.environ["SUPPORTING_PROVIDER"] = "together"
        os.environ.pop("TOGETHER_API_KEY", None)
        os.environ.pop("FIREWORKS_API_KEY", None)
        try:
            with pytest.raises(ValueError, match="No valid API key"):
                get_provider()
        finally:
            del os.environ["SUPPORTING_PROVIDER"]


class TestGetProviderAutoDetect:
    def test_auto_detect_together_only(self):
        os.environ["TOGETHER_API_KEY"] = "sk-test-key-123"
        os.environ.pop("FIREWORKS_API_KEY", None)
        os.environ.pop("SUPPORTING_PROVIDER", None)
        try:
            p = get_provider()
            assert p.name == "together"
        finally:
            del os.environ["TOGETHER_API_KEY"]

    def test_auto_detect_fireworks_only(self):
        os.environ.pop("TOGETHER_API_KEY", None)
        os.environ["FIREWORKS_API_KEY"] = "fw-key-123"
        os.environ.pop("SUPPORTING_PROVIDER", None)
        try:
            p = get_provider()
            assert p.name == "fireworks"
        finally:
            del os.environ["FIREWORKS_API_KEY"]

    def test_auto_detect_both_chooses_first(self):
        """When both keys are set, returns first in dict order."""
        os.environ["TOGETHER_API_KEY"] = "sk-test-key-123"
        os.environ["FIREWORKS_API_KEY"] = "fw-key-123"
        os.environ.pop("SUPPORTING_PROVIDER", None)
        try:
            p = get_provider()
            # Together is first in the dict
            assert p.name == "together"
        finally:
            del os.environ["TOGETHER_API_KEY"]
            del os.environ["FIREWORKS_API_KEY"]

    def test_no_key_at_all_raises(self):
        os.environ.pop("TOGETHER_API_KEY", None)
        os.environ.pop("FIREWORKS_API_KEY", None)
        os.environ.pop("SUPPORTING_PROVIDER", None)
        with pytest.raises(ValueError, match="No provider available"):
            get_provider()
