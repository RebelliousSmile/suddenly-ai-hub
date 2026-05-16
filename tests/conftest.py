"""Fixtures for Suddenly AI Hub tests."""
import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
TEST_PROMPTS = DATA_DIR / "test-prompts.jsonl"


@pytest.fixture
def test_prompts():
    """Load test prompts from JSONL file."""
    prompts = []
    with open(TEST_PROMPTS, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


@pytest.fixture
def test_categories():
    """Expected categories across all test prompts."""
    return {
        "combat_description",
        "combat_dialogue",
        "romance_description",
        "romance_dialogue",
        "intrigue_description",
        "intrigue_dialogue",
        "politique_description",
        "politique_dialogue",
        "quotidien_description",
        "quotidien_dialogue",
    }


@pytest.fixture
def test_universes():
    """Expected universes across all test prompts."""
    return {"fantasy-medievale", "cyberpunk"}


@pytest.fixture
def test_situations():
    """Expected situations across all test prompts."""
    return {"combat", "romance", "intrigue", "politique", "quotidien"}


@pytest.fixture
def test_voices():
    """Expected voices across all test prompts."""
    return {"solennel", "narquois", "theatral", "neutre", "lyrique"}
