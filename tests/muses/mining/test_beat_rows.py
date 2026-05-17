"""Tests de la production de rows niveau `beat`."""

from muses.ingestion.pipeline import TablePaths, ingest
from muses.mining.beats import BEAT_KEYWORDS, build_beat_rows
from muses.tables.embeddings import StubEncoder


def test_build_beat_rows_one_per_known_beat():
    tags = {"univers": ["medieval_fantastique"]}
    rows = build_beat_rows(tags=tags)
    assert len(rows) == len(BEAT_KEYWORDS)
    labels = {r["content"]["label"] for r in rows}
    assert labels == set(BEAT_KEYWORDS.keys())


def test_beat_rows_pass_ingestion(tmp_table_dir):
    tags = {
        "univers": ["medieval_fantastique"],
        "situation": ["combat"],
        "rapport_initial": ["hostile"],
        "voix": ["solennel"],
        "emotion_dominante": ["colere"],
    }
    rows = build_beat_rows(tags=tags)
    paths = TablePaths.from_dir(tmp_table_dir)
    encoder = StubEncoder(dim=8)
    for row_dict in rows:
        result = ingest(row_dict, paths, encoder=encoder, verify_signature=False)
        assert result.success, f"failed: {result.errors}"
