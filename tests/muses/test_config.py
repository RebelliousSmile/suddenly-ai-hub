"""Tests config + entrypoint guard."""

import os

import pytest

from muses.config import ConfigError, load_config


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    """Réinitialise les vars Muses entre les tests."""
    for var in [
        "MUSES_TABLE_DIR", "MUSES_FEEDBACK_DIR", "MUSES_SNAPSHOT_DIR",
        "MUSES_ADMIN_TOKEN", "MUSES_SIGNATURE_MODE", "MUSES_SIGNATURE_MAX_AGE_SECONDS",
        "MUSES_ENCODER", "MUSES_ENCODER_MODEL", "MUSES_STUB_ENCODER_DIM",
        "MUSES_BIND_HOST", "MUSES_BIND_PORT", "MUSES_RATE_LIMIT_PER_MINUTE",
        "MUSES_LOG_LEVEL", "MUSES_LOG_FORMAT",
    ]:
        monkeypatch.delenv(var, raising=False)
    # Minimum requis : table_dir doit exister
    monkeypatch.setenv("MUSES_TABLE_DIR", str(tmp_path / "tables"))
    (tmp_path / "tables").mkdir()
    monkeypatch.setenv("MUSES_FEEDBACK_DIR", str(tmp_path / "feedback"))
    monkeypatch.setenv("MUSES_SNAPSHOT_DIR", str(tmp_path / "snaps"))
    return tmp_path


def test_default_local_bind_no_admin_token_ok(clean_env):
    """Bind 127.0.0.1 sans admin token = OK (dev local)."""
    settings = load_config()
    assert settings.bind_host == "127.0.0.1"
    assert settings.admin_token is None


def test_public_bind_without_admin_token_refused(clean_env, monkeypatch):
    monkeypatch.setenv("MUSES_BIND_HOST", "0.0.0.0")
    with pytest.raises(ConfigError, match="MUSES_ADMIN_TOKEN"):
        load_config()


def test_public_bind_with_admin_token_ok(clean_env, monkeypatch):
    monkeypatch.setenv("MUSES_BIND_HOST", "0.0.0.0")
    monkeypatch.setenv("MUSES_ADMIN_TOKEN", "secret")
    settings = load_config()
    assert settings.bind_host == "0.0.0.0"
    assert settings.admin_token == "secret"


def test_missing_table_dir_refused(clean_env, monkeypatch, tmp_path):
    monkeypatch.setenv("MUSES_TABLE_DIR", str(tmp_path / "absent"))
    with pytest.raises(ConfigError, match="n'existe pas"):
        load_config()


def test_invalid_signature_mode_refused(clean_env, monkeypatch):
    monkeypatch.setenv("MUSES_SIGNATURE_MODE", "weak")
    with pytest.raises(ConfigError, match="MUSES_SIGNATURE_MODE"):
        load_config()


def test_invalid_encoder_refused(clean_env, monkeypatch):
    monkeypatch.setenv("MUSES_ENCODER", "openai")
    with pytest.raises(ConfigError, match="MUSES_ENCODER"):
        load_config()


def test_invalid_int_refused(clean_env, monkeypatch):
    monkeypatch.setenv("MUSES_BIND_PORT", "not-a-number")
    with pytest.raises(ConfigError, match="entier"):
        load_config()


def test_table_jsonl_paths_lists_only_jsonl(clean_env, tmp_path):
    table_dir = tmp_path / "tables"
    (table_dir / "fragments.jsonl").write_text("")
    (table_dir / "beats.jsonl").write_text("")
    (table_dir / "ignored.txt").write_text("")
    (table_dir / "fragments.sqlite").write_text("")
    settings = load_config()
    names = sorted(p.name for p in settings.table_jsonl_paths)
    assert names == ["beats.jsonl", "fragments.jsonl"]
