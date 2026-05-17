"""Tests de l'index SQLite + FTS5."""

import pytest

from muses.schemas.row import Row
from muses.schemas.tags import AxialTags
from muses.tables.jsonl_io import append_row
from muses.tables.sqlite_index import (
    create_schema,
    query_by_tags,
    reindex_from_jsonl,
    search_fts,
    upsert_row,
)


def _frag(text: str, univers="medieval_fantastique", situation=None, voix=None) -> Row:
    return Row(
        level="fragment",
        tags=AxialTags(
            univers=[univers],
            situation=[situation] if situation else [],
            voix=[voix] if voix else [],
        ),
        content={"text": text},
        source="bootstrap",
    )


def test_create_schema_idempotent(tmp_path):
    db = tmp_path / "x.sqlite"
    create_schema(db)
    create_schema(db)  # ne doit pas lever
    assert db.exists()


def test_upsert_and_query_by_tags(tmp_path):
    db = tmp_path / "x.sqlite"
    jsonl = tmp_path / "x.jsonl"
    create_schema(db)
    r1 = _frag("medieval combat", univers="medieval_fantastique", situation="combat")
    r2 = _frag("cyberpunk hack", univers="cyberpunk", situation="combat")
    upsert_row(db, r1, jsonl)
    upsert_row(db, r2, jsonl)

    ids = query_by_tags(db, tags={"univers": ["medieval_fantastique"]})
    assert ids == [r1.id]

    ids = query_by_tags(db, tags={"situation": ["combat"]})
    assert set(ids) == {r1.id, r2.id}


def test_query_universal_row_matches_anything(tmp_path):
    db = tmp_path / "x.sqlite"
    jsonl = tmp_path / "x.jsonl"
    create_schema(db)
    universal = Row(
        level="fragment",
        tags=AxialTags(),  # toutes listes vides
        content={"text": "universel"},
        source="bootstrap",
    )
    upsert_row(db, universal, jsonl)

    ids = query_by_tags(db, tags={"univers": ["cyberpunk"]})
    assert ids == [universal.id]


def test_reindex_from_jsonl(tmp_path):
    jsonl = tmp_path / "t.jsonl"
    db = tmp_path / "t.sqlite"
    append_row(jsonl, _frag("a"))
    append_row(jsonl, _frag("b"))
    append_row(jsonl, _frag("c"))
    count = reindex_from_jsonl(db, jsonl)
    assert count == 3


def test_search_fts(tmp_path):
    db = tmp_path / "x.sqlite"
    jsonl = tmp_path / "x.jsonl"
    create_schema(db)
    r1 = _frag("le marchand cynique sort son épée")
    r2 = _frag("la pluie tombe sur les ruines")
    upsert_row(db, r1, jsonl)
    upsert_row(db, r2, jsonl)

    ids = search_fts(db, "marchand")
    assert r1.id in ids
    assert r2.id not in ids


def test_archived_excluded_from_query(tmp_path):
    from datetime import datetime, timezone

    db = tmp_path / "x.sqlite"
    jsonl = tmp_path / "x.jsonl"
    create_schema(db)
    archived = Row(
        level="fragment",
        tags=AxialTags(univers=["medieval_fantastique"]),
        content={"text": "vieux fragment"},
        source="bootstrap",
        archived_at=datetime.now(tz=timezone.utc),
    )
    upsert_row(db, archived, jsonl)

    ids = query_by_tags(db, tags={"univers": ["medieval_fantastique"]})
    assert ids == []
    ids_with_archived = query_by_tags(
        db, tags={"univers": ["medieval_fantastique"]}, include_archived=True
    )
    assert ids_with_archived == [archived.id]
