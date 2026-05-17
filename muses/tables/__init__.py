"""I/O des tables Muses : JSONL, index SQLite FTS5, embeddings npy."""

from muses.tables.jsonl_io import append_row, count_rows, iter_rows, read_rows
from muses.tables.sqlite_index import (
    create_schema,
    reindex_from_jsonl,
    upsert_row,
    query_by_tags,
    search_fts,
)
from muses.tables.embeddings import (
    EmbeddingsCache,
    StubEncoder,
    rebuild_for_table,
)

__all__ = [
    "append_row",
    "count_rows",
    "iter_rows",
    "read_rows",
    "create_schema",
    "reindex_from_jsonl",
    "upsert_row",
    "query_by_tags",
    "search_fts",
    "EmbeddingsCache",
    "StubEncoder",
    "rebuild_for_table",
]
