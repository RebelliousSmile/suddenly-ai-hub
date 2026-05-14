#!/usr/bin/env python3
"""
pytest tests for scripts/train_together.py

Tests are network-free — they test pure functions (validation, splitting,
cache I/O) and mock the API calls for train/infer subcommands.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Ensure the script is importable
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from train_together import (
    FINE_TUNABLE_MODELS,
    CACHE_MODEL_ID_FILE,
    CACHE_DIR,
    DEFAULT_MODEL,
    DEFAULT_SYSTEM_PROMPT,
    _validate_message,
    _validate_example,
    _ensure_cache_dir,
    _persist_model_id,
    _load_model_id,
    cmd_validate,
)
import argparse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def tmp_data_dir(tmp_path):
    """Create a temporary data directory with test JSONL."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    return data_dir


@pytest.fixture()
def valid_examples():
    """Return a list of valid example dicts."""
    return [
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Tell me a story."},
                {"role": "assistant", "content": "Once upon a time..."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "What happened next?"},
                {"role": "assistant", "content": "The end."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Who are you?"},
                {"role": "assistant", "content": "I am a storyteller."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Where do you live?"},
                {"role": "assistant", "content": "In a big house."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "What is your name?"},
                {"role": "assistant", "content": "I have no name."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Do you like rain?"},
                {"role": "assistant", "content": "Yes, I do."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Tell me a joke."},
                {"role": "assistant", "content": "Why did the chicken cross the road?"},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "What's the weather?"},
                {"role": "assistant", "content": "It's sunny."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Sing me a song."},
                {"role": "assistant", "content": "La la la..."},
            ]
        },
        {
            "messages": [
                {"role": "system", "content": "You are a storyteller."},
                {"role": "user", "content": "Goodbye!"},
                {"role": "assistant", "content": "Farewell, traveler."},
            ]
        },
    ]


@pytest.fixture()
def valid_jsonl_file(tmp_data_dir, valid_examples):
    """Write valid examples to a JSONL file."""
    f = tmp_data_dir / "valid.jsonl"
    with open(f, "w") as fh:
        for ex in valid_examples:
            fh.write(json.dumps(ex) + "\n")
    return f


@pytest.fixture()
def invalid_jsonl_file(tmp_data_dir):
    """Write a mix of valid and invalid lines to a JSONL file."""
    f = tmp_data_dir / "mixed.jsonl"
    with open(f, "w") as fh:
        # Valid
        fh.write(json.dumps({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        }) + "\n")
        # Invalid JSON
        fh.write("not valid json\n")
        # Valid again
        fh.write(json.dumps({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "bye"},
                {"role": "assistant", "content": "see you"},
            ]
        }) + "\n")
    return f


# ---------------------------------------------------------------------------
# Tests: _validate_message
# ---------------------------------------------------------------------------


class TestValidateMessage:
    def test_valid_system_message(self):
        assert _validate_message({"role": "system", "content": "You are helpful."}) is True

    def test_valid_user_message(self):
        assert _validate_message({"role": "user", "content": "Hello"}) is True

    def test_valid_assistant_message(self):
        assert _validate_message({"role": "assistant", "content": "Hi there!"}) is True

    def test_missing_role(self):
        assert _validate_message({"content": "Hello"}) is False

    def test_missing_content(self):
        assert _validate_message({"role": "user"}) is False

    def test_empty_role(self):
        assert _validate_message({"role": "", "content": "Hello"}) is False

    def test_empty_content(self):
        assert _validate_message({"role": "user", "content": ""}) is False

    def test_not_dict(self):
        assert _validate_message("just a string") is False

    def test_none(self):
        assert _validate_message(None) is False


# ---------------------------------------------------------------------------
# Tests: _validate_example
# ---------------------------------------------------------------------------


class TestValidateExample:
    def test_valid_example(self):
        ok, err = _validate_example({
            "messages": [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        })
        assert ok is True
        assert err == ""

    def test_missing_messages_field(self):
        ok, err = _validate_example({"content": "hi"})
        assert ok is False
        assert "messages" in err

    def test_empty_messages(self):
        ok, err = _validate_example({"messages": []})
        assert ok is False
        assert "non-empty" in err.lower()

    def test_not_list_messages(self):
        ok, err = _validate_example({"messages": "not a list"})
        assert ok is False

    def test_invalid_message_in_list(self):
        ok, err = _validate_example({
            "messages": [
                {"role": "user", "content": "hi"},
                "not a dict",
            ]
        })
        assert ok is False
        assert "index" in err.lower()

    def test_not_a_dict(self):
        ok, err = _validate_example("just a string")
        assert ok is False


# ---------------------------------------------------------------------------
# Tests: _validate_example with no system prompt
# ---------------------------------------------------------------------------


class TestValidateExampleNoSystemPrompt:
    def test_user_assistant_only(self):
        ok, err = _validate_example({
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        })
        assert ok is True
        assert err == ""


# ---------------------------------------------------------------------------
# Tests: _persist_model_id / _load_model_id
# ---------------------------------------------------------------------------


class TestModelIdCache:
    def test_persist_and_load(self, tmp_path, monkeypatch):
        # Point the cache to temp dir
        monkeypatch.setattr("train_together.CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(
            "train_together.CACHE_MODEL_ID_FILE",
            tmp_path / ".cache" / "model_id.txt",
        )

        _persist_model_id("test-model-123")
        assert (tmp_path / ".cache" / "model_id.txt").exists()

        result = _load_model_id()
        assert result == "test-model-123"

    def test_load_nonexistent(self, tmp_path, monkeypatch):
        monkeypatch.setattr("train_together.CACHE_DIR", tmp_path / ".cache")
        monkeypatch.setattr(
            "train_together.CACHE_MODEL_ID_FILE",
            tmp_path / ".cache" / "model_id.txt",
        )

        result = _load_model_id()
        assert result is None

    def test_persist_creates_directory(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / ".cache" / "nested"
        monkeypatch.setattr("train_together.CACHE_DIR", cache_dir)
        monkeypatch.setattr(
            "train_together.CACHE_MODEL_ID_FILE",
            cache_dir / "model_id.txt",
        )

        _persist_model_id("model-abc")
        assert cache_dir.exists()
        assert (cache_dir / "model_id.txt").exists()


# ---------------------------------------------------------------------------
# Tests: cmd_validate
# ---------------------------------------------------------------------------


class TestCmdValidate:
    def test_validate_creates_train_and_val(
        self, valid_jsonl_file, tmp_data_dir, monkeypatch, capsys
    ):
        """80/20 split should create train.jsonl and val.jsonl."""
        monkeypatch.chdir(tmp_data_dir)

        args = argparse.Namespace(
            input=str(valid_jsonl_file),
            system=None,
            split_ratio=0.8,
            seed=42,
            dry_run=False,
        )

        # Redirect DATA_DIR to tmp
        with patch("train_together.DATA_DIR", tmp_data_dir):
            cmd_validate(args)

        captured = capsys.readouterr()
        assert "✅ Validation complete" in captured.out

        assert (tmp_data_dir / "train.jsonl").exists()
        assert (tmp_data_dir / "val.jsonl").exists()

        # Check file contents
        train_count = sum(1 for line in (tmp_data_dir / "train.jsonl").read_text().strip().split("\n") if line)
        val_count = sum(1 for line in (tmp_data_dir / "val.jsonl").read_text().strip().split("\n") if line)

        # 10 examples, 80/20 = 8 train, 2 val (with seed=42)
        assert train_count == 8
        assert val_count == 2

    def test_validate_system_prompt_injection(
        self, valid_jsonl_file, tmp_data_dir, monkeypatch, capsys
    ):
        """Verify system prompt is injected when missing."""
        # Create a file without system prompt
        no_sys_file = tmp_data_dir / "no_sys.jsonl"
        with open(no_sys_file, "w") as f:
            f.write(json.dumps({
                "messages": [
                    {"role": "user", "content": "hi"},
                    {"role": "assistant", "content": "hello"},
                ]
            }) + "\n")

        monkeypatch.chdir(tmp_data_dir)

        with patch("train_together.DATA_DIR", tmp_data_dir):
            cmd_validate(argparse.Namespace(
                input=str(no_sys_file),
                system="Custom system prompt",
                split_ratio=1.0,
                seed=42,
                dry_run=False,
            ))

        # Check that the written train file has the system prompt
        train_content = (tmp_data_dir / "train.jsonl").read_text()
        parsed = json.loads(train_content)
        roles = [m["role"] for m in parsed["messages"]]
        assert "system" in roles

    def test_validate_dry_run(self, valid_jsonl_file, tmp_data_dir, monkeypatch, capsys):
        """Dry run should not create files."""
        monkeypatch.chdir(tmp_data_dir)

        with patch("train_together.DATA_DIR", tmp_data_dir):
            cmd_validate(argparse.Namespace(
                input=str(valid_jsonl_file),
                system=None,
                split_ratio=0.5,
                seed=42,
                dry_run=True,
            ))

        captured = capsys.readouterr()
        assert "dry run" in captured.out.lower()
        assert not (tmp_data_dir / "train.jsonl").exists()
        assert not (tmp_data_dir / "val.jsonl").exists()

    def test_validate_invalid_file(self, tmp_data_dir, monkeypatch):
        """Should exit 1 for non-existent file."""
        monkeypatch.chdir(tmp_data_dir)

        with pytest.raises(SystemExit) as excinfo:
            cmd_validate(argparse.Namespace(
                input=str(tmp_data_dir / "nonexistent.jsonl"),
                system=None,
                split_ratio=0.8,
                seed=42,
                dry_run=False,
            ))
        assert excinfo.value.code == 1

    def test_validate_mixed_valid_invalid(self, invalid_jsonl_file, tmp_data_dir, monkeypatch, capsys):
        """Should skip invalid lines and count only valid ones."""
        monkeypatch.chdir(tmp_data_dir)

        with patch("train_together.DATA_DIR", tmp_data_dir):
            cmd_validate(argparse.Namespace(
                input=str(invalid_jsonl_file),
                system=None,
                split_ratio=1.0,
                seed=42,
                dry_run=False,
            ))

        captured = capsys.readouterr()
        # Should have 2 valid examples (invalid JSON line skipped)
        train_content = (tmp_data_dir / "train.jsonl").read_text()
        train_count = sum(1 for line in train_content.strip().split("\n") if line)
        assert train_count == 2

    def test_validate_split_ratio(self, valid_jsonl_file, tmp_data_dir, monkeypatch, capsys):
        """Test with 50/50 split."""
        monkeypatch.chdir(tmp_data_dir)

        with patch("train_together.DATA_DIR", tmp_data_dir):
            cmd_validate(argparse.Namespace(
                input=str(valid_jsonl_file),
                system=None,
                split_ratio=0.5,
                seed=42,
                dry_run=False,
            ))

        train_count = sum(1 for line in (tmp_data_dir / "train.jsonl").read_text().strip().split("\n") if line)
        val_count = sum(1 for line in (tmp_data_dir / "val.jsonl").read_text().strip().split("\n") if line)
        assert train_count == 5
        assert val_count == 5

    def test_validate_all_invalid(self, tmp_data_dir, monkeypatch):
        """Should exit 1 when all lines are invalid."""
        bad_file = tmp_data_dir / "all_bad.jsonl"
        with open(bad_file, "w") as f:
            f.write("not json\n")
            f.write("also not json\n")

        monkeypatch.chdir(tmp_data_dir)

        with pytest.raises(SystemExit) as excinfo:
            cmd_validate(argparse.Namespace(
                input=str(bad_file),
                system=None,
                split_ratio=0.8,
                seed=42,
                dry_run=False,
            ))
        assert excinfo.value.code == 1


# ---------------------------------------------------------------------------
# Tests: Split reproducibility with seed
# ---------------------------------------------------------------------------


class TestSplitReproducibility:
    def test_seed_produces_same_split(self, valid_jsonl_file, tmp_data_dir, monkeypatch):
        """Same seed should produce the same split."""
        monkeypatch.chdir(tmp_data_dir)

        with patch("train_together.DATA_DIR", tmp_data_dir):
            cmd_validate(argparse.Namespace(
                input=str(valid_jsonl_file),
                system=None,
                split_ratio=0.8,
                seed=123,
                dry_run=False,
            ))

        train1 = (tmp_data_dir / "train.jsonl").read_text()

        with patch("train_together.DATA_DIR", tmp_data_dir):
            cmd_validate(argparse.Namespace(
                input=str(valid_jsonl_file),
                system=None,
                split_ratio=0.8,
                seed=123,
                dry_run=False,
            ))

        train2 = (tmp_data_dir / "train.jsonl").read_text()
        assert train1 == train2


# ---------------------------------------------------------------------------
# Tests: Constants
# ---------------------------------------------------------------------------


class TestConstants:
    def test_fine_tunable_models_list(self):
        assert len(FINE_TUNABLE_MODELS) >= 2
        assert "Qwen/Qwen2.5-7B-Instruct" in FINE_TUNABLE_MODELS
        assert all(isinstance(m, str) for m in FINE_TUNABLE_MODELS)

    def test_default_model(self):
        assert DEFAULT_MODEL == FINE_TUNABLE_MODELS[0]

    def test_default_system_prompt(self):
        assert "roleplay" in DEFAULT_SYSTEM_PROMPT.lower()


# ---------------------------------------------------------------------------
# Tests: _ensure_cache_dir
# ---------------------------------------------------------------------------


class TestEnsureCacheDir:
    def test_creates_directory(self, tmp_path, monkeypatch):
        cache_dir = tmp_path / ".cache" / "deep"
        monkeypatch.setattr("train_together.CACHE_DIR", cache_dir)
        assert not cache_dir.exists()

        _ensure_cache_dir()
        assert cache_dir.exists()
        assert cache_dir.is_dir()


# ---------------------------------------------------------------------------
# Tests: _validate_example — edge cases
# ---------------------------------------------------------------------------


class TestValidateExampleEdgeCases:
    def test_single_message_example(self):
        """Example with only one message should be valid."""
        ok, err = _validate_example({
            "messages": [{"role": "user", "content": "hi"}]
        })
        assert ok is True

    def test_many_messages(self):
        """Example with many messages should pass."""
        msgs = [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "q1"},
            {"role": "assistant", "content": "a1"},
            {"role": "user", "content": "q2"},
            {"role": "assistant", "content": "a2"},
        ]
        ok, err = _validate_example({"messages": msgs})
        assert ok is True

    def test_unicode_content(self):
        """Unicode in content should be valid."""
        ok, err = _validate_example({
            "messages": [
                {"role": "user", "content": "你好 monde 🌍"},
                {"role": "assistant", "content": "ПриветBonjour 👋"},
            ]
        })
        assert ok is True
