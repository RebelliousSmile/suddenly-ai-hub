"""Tests du pipeline d'ingestion end-to-end (M0)."""

import numpy as np
import pytest

from muses.ingestion.pipeline import IngestionResult, TablePaths, ingest
from muses.tables.embeddings import EmbeddingsCache, StubEncoder
from muses.tables.jsonl_io import count_rows
from muses.tables.sqlite_index import query_by_tags


def _bootstrap_row_data(text: str = "« Hello »") -> dict:
    return {
        "level": "fragment",
        "tags": {
            "univers": ["medieval_fantastique"],
            "situation": ["combat"],
            "rapport_initial": ["hostile"],
            "voix": ["narquois"],
            "emotion_dominante": ["colere"],
        },
        "content": {"text": text, "char_pov": "neutral"},
        "source": "bootstrap",
    }


def _contribution_row_data(text: str = "« Hello »") -> dict:
    return {
        "level": "fragment",
        "tags": {
            "univers": ["medieval_fantastique"],
            "situation": ["combat"],
            "rapport_initial": ["hostile"],
            "voix": ["narquois"],
            "emotion_dominante": ["colere"],
        },
        "content": {"text": text, "char_pov": "neutral"},
        "source": "contribution_explicit",
        "user_id": "https://exemple.tld/users/alice",
        "instance_id": "exemple.tld",
        "signature": "keyId=\"alice\",algorithm=\"rsa-sha256\",signature=\"...\"",
    }


def test_ingest_bootstrap_row_success(tmp_table_dir):
    paths = TablePaths.from_dir(tmp_table_dir)
    encoder = StubEncoder(dim=8)

    result = ingest(_bootstrap_row_data(), paths, encoder=encoder)

    assert result.success is True
    assert result.row_id is not None
    assert result.errors == []
    assert count_rows(paths.jsonl) == 1
    assert EmbeddingsCache(paths.npy).load().shape == (1, 8)


def test_ingest_contribution_row_success(tmp_table_dir):
    paths = TablePaths.from_dir(tmp_table_dir)
    result = ingest(_contribution_row_data(), paths, encoder=StubEncoder(dim=8))
    assert result.success is True


def test_ingest_rejects_invalid_schema(tmp_table_dir):
    paths = TablePaths.from_dir(tmp_table_dir)
    bad = {"level": "fragment", "content": {"text": "x"}}  # missing tags, source
    result = ingest(bad, paths, encoder=StubEncoder(dim=8))
    assert result.success is False
    assert result.stage_failed == "schema_validation"


def test_ingest_rejects_invalid_tag_value(tmp_table_dir):
    paths = TablePaths.from_dir(tmp_table_dir)
    data = _bootstrap_row_data()
    data["tags"]["univers"] = ["medieval-fantastique"]  # hyphen invalide
    result = ingest(data, paths, encoder=StubEncoder(dim=8))
    assert result.success is False
    assert result.stage_failed == "schema_validation"


def test_ingest_rejects_content_mismatched_level(tmp_table_dir):
    paths = TablePaths.from_dir(tmp_table_dir)
    data = _bootstrap_row_data()
    data["level"] = "entity"  # mais content reste un fragment
    result = ingest(data, paths, encoder=StubEncoder(dim=8))
    assert result.success is False
    assert result.stage_failed == "schema_validation"


def test_ingest_rejects_missing_signature_for_contribution(tmp_table_dir):
    paths = TablePaths.from_dir(tmp_table_dir)
    data = _contribution_row_data()
    data["signature"] = None
    result = ingest(data, paths, encoder=StubEncoder(dim=8))
    # Le schéma Row valide déjà l'absence de signature pour contribution → rejet schéma
    assert result.success is False


def test_ingest_multiple_rows_keeps_alignment(tmp_table_dir):
    paths = TablePaths.from_dir(tmp_table_dir)
    encoder = StubEncoder(dim=8)
    for i in range(5):
        data = _bootstrap_row_data(text=f"row {i}")
        result = ingest(data, paths, encoder=encoder)
        assert result.success is True

    assert count_rows(paths.jsonl) == 5
    embeddings = EmbeddingsCache(paths.npy).load()
    assert embeddings.shape == (5, 8)
    # Toutes les rows sont trouvables par leur tag
    ids = query_by_tags(paths.db, tags={"univers": ["medieval_fantastique"]})
    assert len(ids) == 5
