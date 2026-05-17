"""Tests de l'extracteur d'entités lexicon-based."""

from muses.ingestion.pipeline import TablePaths, ingest
from muses.mining.entities import LEXICON, build_entity_rows
from muses.tables.embeddings import StubEncoder


def test_lexicon_non_empty():
    assert LEXICON
    for ent_type, entries in LEXICON.items():
        assert entries, f"type {ent_type} a un lexique vide"


def test_build_entity_rows_produces_correct_count():
    tags = {"univers": ["medieval_fantastique"]}
    rows = build_entity_rows(tags=tags, source="bootstrap")
    expected = sum(len(entries) for entries in LEXICON.values())
    assert len(rows) == expected


def test_entity_rows_carry_required_fields():
    tags = {"univers": ["medieval_fantastique"]}
    rows = build_entity_rows(tags=tags, source="bootstrap")
    for r in rows:
        assert r["level"] == "entity"
        assert "type" in r["content"]
        assert "lemma" in r["content"]
        assert "forms" in r["content"]
        assert r["tags"] == tags


def test_entity_rows_pass_ingestion(tmp_table_dir):
    tags = {
        "univers": ["medieval_fantastique"],
        "situation": ["combat"],
        "rapport_initial": ["hostile"],
        "voix": ["solennel"],
        "emotion_dominante": ["colere"],
    }
    rows = build_entity_rows(tags=tags, source="bootstrap")
    paths = TablePaths.from_dir(tmp_table_dir)
    encoder = StubEncoder(dim=8)
    for row_dict in rows:
        result = ingest(row_dict, paths, encoder=encoder, verify_signature=False)
        assert result.success, f"failed: {result.errors}"
