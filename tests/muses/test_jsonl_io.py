"""Tests I/O JSONL."""

from muses.schemas.row import Row
from muses.schemas.tags import AxialTags
from muses.tables.jsonl_io import append_row, count_rows, iter_rows, read_rows


def _frag(text: str) -> Row:
    return Row(
        level="fragment",
        tags=AxialTags(univers=["medieval_fantastique"]),
        content={"text": text},
        source="bootstrap",
    )


def test_append_and_count(tmp_path):
    path = tmp_path / "t.jsonl"
    assert count_rows(path) == 0
    append_row(path, _frag("a"))
    assert count_rows(path) == 1
    append_row(path, _frag("b"))
    assert count_rows(path) == 2


def test_iter_preserves_order(tmp_path):
    path = tmp_path / "t.jsonl"
    texts = ["one", "two", "three"]
    for t in texts:
        append_row(path, _frag(t))
    read_texts = [r.parsed_content().text for r in iter_rows(path)]
    assert read_texts == texts


def test_read_rows_round_trip(tmp_path):
    path = tmp_path / "t.jsonl"
    original = _frag("« Hello »")
    append_row(path, original)
    rows = read_rows(path)
    assert len(rows) == 1
    assert rows[0].id == original.id
    assert rows[0].parsed_content().text == "« Hello »"


def test_missing_file_returns_empty(tmp_path):
    path = tmp_path / "absent.jsonl"
    assert count_rows(path) == 0
    assert read_rows(path) == []


def test_creates_parent_directory(tmp_path):
    path = tmp_path / "nested" / "deep" / "t.jsonl"
    append_row(path, _frag("x"))
    assert path.exists()
    assert count_rows(path) == 1


def test_blank_lines_skipped(tmp_path):
    path = tmp_path / "t.jsonl"
    row = _frag("only one")
    append_row(path, row)
    # Inject blank lines manually
    with path.open("a", encoding="utf-8") as fh:
        fh.write("\n\n   \n")
    assert count_rows(path) == 1
    assert len(read_rows(path)) == 1
