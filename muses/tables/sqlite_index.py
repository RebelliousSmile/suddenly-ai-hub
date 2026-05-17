"""Index SQLite + FTS5 pour query rapide sur les tables Muses.

L'index est **reconstruit** depuis les JSONL — il n'est pas la source de vérité.
Le détail du schéma est dans data-format.md §Index SQLite.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from muses.schemas.row import Row
from muses.schemas.tags import AXIS_NAMES
from muses.tables.jsonl_io import iter_rows


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS rows (
    id TEXT PRIMARY KEY,
    table_path TEXT NOT NULL,
    level TEXT NOT NULL,
    user_id TEXT,
    instance_id TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    quality_score REAL,
    archived_at TEXT,
    tag_univers TEXT,
    tag_situation TEXT,
    tag_rapport_initial TEXT,
    tag_voix TEXT,
    tag_emotion_dominante TEXT,
    content_text TEXT
);

CREATE INDEX IF NOT EXISTS idx_rows_level ON rows(level);
CREATE INDEX IF NOT EXISTS idx_rows_source ON rows(source);
CREATE INDEX IF NOT EXISTS idx_rows_archived ON rows(archived_at);

CREATE VIRTUAL TABLE IF NOT EXISTS rows_fts USING fts5(
    content_text,
    content='rows',
    content_rowid='rowid'
);
"""


class FTS5NotAvailable(RuntimeError):
    """SQLite local ne supporte pas FTS5 (rare sur CPython moderne mais possible)."""


def _check_fts5_available(conn: sqlite3.Connection) -> None:
    """Vérifie que FTS5 est compilé dans la lib SQLite courante."""
    try:
        conn.execute("CREATE VIRTUAL TABLE _fts5_probe USING fts5(x)")
        conn.execute("DROP TABLE _fts5_probe")
    except sqlite3.OperationalError as exc:
        raise FTS5NotAvailable(
            "Cette installation Python n'a pas SQLite avec FTS5 activé. "
            "Solution typique : reinstaller Python avec sqlite3 plus récent, "
            "ou utiliser pysqlite3-binary."
        ) from exc


def create_schema(db_path: Path) -> None:
    """Crée les tables et index si absents. Vérifie aussi le support FTS5."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    with sqlite3.connect(db_path) as conn:
        _check_fts5_available(conn)
        conn.executescript(SCHEMA_SQL)


def upsert_row(db_path: Path, row: Row, table_path: Path) -> None:
    """Insère ou met à jour une row dans l'index. Met aussi à jour FTS."""
    db_path = Path(db_path)
    table_path = Path(table_path)
    content_text = row.embeddable_text()

    with sqlite3.connect(db_path) as conn:
        cur = conn.cursor()
        cur.execute(
            """
            INSERT OR REPLACE INTO rows (
                id, table_path, level, user_id, instance_id, source,
                created_at, quality_score, archived_at,
                tag_univers, tag_situation, tag_rapport_initial, tag_voix, tag_emotion_dominante,
                content_text
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                row.id,
                str(table_path),
                row.level,
                row.user_id,
                row.instance_id,
                row.source,
                row.created_at.isoformat(),
                row.quality_score,
                row.archived_at.isoformat() if row.archived_at else None,
                json.dumps(row.tags.univers),
                json.dumps(row.tags.situation),
                json.dumps(row.tags.rapport_initial),
                json.dumps(row.tags.voix),
                json.dumps(row.tags.emotion_dominante),
                content_text,
            ),
        )
        # Sync FTS5 (rebuild les lignes touchées)
        rowid = cur.execute(
            "SELECT rowid FROM rows WHERE id = ?", (row.id,)
        ).fetchone()[0]
        cur.execute("INSERT OR REPLACE INTO rows_fts(rowid, content_text) VALUES (?, ?)",
                    (rowid, content_text))
        conn.commit()


def reindex_from_jsonl(db_path: Path, jsonl_path: Path) -> int:
    """Reconstruit l'index pour une table depuis son JSONL. Renvoie le nombre de rows."""
    create_schema(db_path)
    count = 0
    for row in iter_rows(jsonl_path):
        upsert_row(db_path, row, jsonl_path)
        count += 1
    return count


def query_by_tags(
    db_path: Path,
    *,
    tags: dict[str, list[str]] | None = None,
    level: str | None = None,
    include_archived: bool = False,
) -> list[str]:
    """Renvoie les ids de rows compatibles avec les tags fournis.

    Compatibilité par axe : la row matche si elle n'a pas de tag sur cet axe
    (universelle) ou si son tag intersecte avec celui demandé.
    """
    where_clauses = []
    params: list = []

    if level is not None:
        where_clauses.append("level = ?")
        params.append(level)
    if not include_archived:
        where_clauses.append("archived_at IS NULL")

    sql = "SELECT id, " + ", ".join(f"tag_{a}" for a in AXIS_NAMES) + " FROM rows"
    if where_clauses:
        sql += " WHERE " + " AND ".join(where_clauses)

    matching_ids: list[str] = []
    with sqlite3.connect(db_path) as conn:
        for row_data in conn.execute(sql, params):
            row_id = row_data[0]
            row_tags = {
                axis: json.loads(row_data[1 + idx]) if row_data[1 + idx] else []
                for idx, axis in enumerate(AXIS_NAMES)
            }
            if _tags_compatible(row_tags, tags or {}):
                matching_ids.append(row_id)
    return matching_ids


def _tags_compatible(row_tags: dict[str, list[str]], ctx_tags: dict[str, list[str]]) -> bool:
    """Variante dict de AxialTags.is_compatible_with (cf. schemas/tags.py).

    Volontairement dupliquée pour éviter de construire un AxialTags Pydantic
    pour chaque row indexée — l'index SQLite manipule directement des dicts
    désérialisés depuis les colonnes JSON, c'est le chemin chaud des queries.
    Les deux implémentations doivent rester sémantiquement équivalentes.
    """
    for axis in AXIS_NAMES:
        row_values = set(row_tags.get(axis, []))
        if not row_values:
            continue
        ctx_values = set(ctx_tags.get(axis, []))
        if not ctx_values:
            continue
        if row_values.isdisjoint(ctx_values):
            return False
    return True


def search_fts(db_path: Path, query_text: str, limit: int = 50) -> list[str]:
    """Recherche full-text sur le content. Renvoie les ids ordonnés par pertinence."""
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            """
            SELECT r.id
            FROM rows_fts
            JOIN rows r ON r.rowid = rows_fts.rowid
            WHERE rows_fts MATCH ?
              AND r.archived_at IS NULL
            ORDER BY rank
            LIMIT ?
            """,
            (query_text, limit),
        ).fetchall()
    return [row[0] for row in rows]
