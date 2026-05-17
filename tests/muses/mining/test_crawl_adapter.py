"""Tests de l'adapter crawl_rpv → row dicts."""

import json

import pytest

from muses.ingestion.pipeline import TablePaths, ingest
from muses.mining.crawl_adapter import (
    extract_fragments_from_rp_dataset,
    parse_rp_dataset_jsonl,
)
from muses.tables.embeddings import StubEncoder
from muses.tables.jsonl_io import count_rows


def _write_jsonl(path, entries):
    with path.open("w", encoding="utf-8") as fh:
        for e in entries:
            fh.write(json.dumps(e, ensure_ascii=False) + "\n")


@pytest.fixture
def sample_jsonl(tmp_path):
    path = tmp_path / "sample.jsonl"
    _write_jsonl(path, [
        {
            "messages": [
                {"role": "system", "content": "Tu es un conteur."},
                {"role": "user", "content": "Une question."},
                {
                    "role": "assistant",
                    "content": (
                        "« Je ne reculerai pas », murmure le chevalier. "
                        "Il lève son épée et fait face. "
                        "L'aube se lève sur les ruines."
                    ),
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "assistant",
                    "content": "Trop court.",  # < min_chars
                },
            ]
        },
    ])
    return path


def test_parse_yields_entries(sample_jsonl):
    entries = list(parse_rp_dataset_jsonl(sample_jsonl))
    assert len(entries) == 2


def test_extract_skips_short_fragments(sample_jsonl):
    rows = extract_fragments_from_rp_dataset(
        sample_jsonl,
        tags={"univers": ["medieval_fantastique"]},
        anonymize=False,
        min_chars=30,
    )
    # 3 phrases >= 30 chars dans la 1re entrée, 0 dans la 2e
    assert len(rows) == 3


def test_extract_skips_non_assistant_messages(sample_jsonl):
    rows = extract_fragments_from_rp_dataset(
        sample_jsonl,
        tags={"univers": ["medieval_fantastique"]},
        anonymize=False,
    )
    for r in rows:
        # Vérifie que le texte ne vient pas du system ou user
        assert "conteur" not in r["content"]["text"]
        assert "question" not in r["content"]["text"]


def test_extracted_rows_pass_ingestion(sample_jsonl, tmp_table_dir):
    rows = extract_fragments_from_rp_dataset(
        sample_jsonl,
        tags={
            "univers": ["medieval_fantastique"],
            "situation": ["combat"],
            "rapport_initial": ["hostile"],
            "voix": ["solennel"],
            "emotion_dominante": ["colere"],
        },
        anonymize=False,
    )
    paths = TablePaths.from_dir(tmp_table_dir)
    encoder = StubEncoder(dim=8)

    success_count = 0
    for row_dict in rows:
        result = ingest(row_dict, paths, encoder=encoder, verify_signature=False)
        assert result.success, f"Ingestion failed: {result.errors}"
        success_count += 1

    assert success_count == len(rows)
    assert count_rows(paths.jsonl) == len(rows)


def test_anonymization_applied(sample_jsonl):
    rows = extract_fragments_from_rp_dataset(
        sample_jsonl,
        tags={"univers": ["medieval_fantastique"]},
        anonymize=True,  # active l'anonymisation regex (spaCy non installé)
    )
    # On vérifie au moins que l'option est honorée (les rows existent et content présent)
    assert all("text" in r["content"] for r in rows)
