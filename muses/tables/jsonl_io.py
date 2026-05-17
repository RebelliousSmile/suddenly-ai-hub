"""I/O JSONL append-only sur les tables Muses.

Une row par ligne. Pas de suppression in-place (l'archivage se fait par flag
`archived_at` dans la row elle-même — cf. data-format.md).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterator

from muses.schemas.row import Row


def append_row(path: Path, row: Row) -> None:
    """Ajoute une row à la fin du JSONL. Crée le fichier si absent.

    L'ordre d'insertion est stable : il définit le mapping `id ↔ index`
    utilisé pour aligner les embeddings .npy (cf. data-format.md §Embeddings).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(row.model_dump_json() + "\n")


def iter_rows(path: Path) -> Iterator[Row]:
    """Itère sur les rows du JSONL. Lazy, utile pour les grandes tables."""
    path = Path(path)
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield Row.model_validate_json(line)


def read_rows(path: Path) -> list[Row]:
    """Lit toutes les rows en mémoire. Pour les petites tables ou les tests."""
    return list(iter_rows(path))


def count_rows(path: Path) -> int:
    """Compte les rows non vides du fichier sans tout parser."""
    path = Path(path)
    if not path.exists():
        return 0
    count = 0
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count
