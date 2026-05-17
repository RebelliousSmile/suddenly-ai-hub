"""Pytest fixtures partagés."""
from pathlib import Path

import pytest


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


@pytest.fixture
def tmp_table_dir(tmp_path):
    """Dossier temporaire pour une table de test (JSONL + SQLite + npy)."""
    table_dir = tmp_path / "table"
    table_dir.mkdir()
    return table_dir
