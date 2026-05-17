"""Pipeline d'ingestion d'une row vers une table Muses.

Étapes pour le MVP M0 (cf. external/data-format.md § Validation à l'ingestion) :

1. Validation du schéma (Pydantic Row)
2. Validation des tags (set canonique)
3. Vérification de signature ActivityPub — **stub M0**, vraie vérif en M2/T22
4. Anonymisation — **assumée déjà faite par le caller en M0** ; l'intégration de
   `pipelines/anonymization/` est M1/T11
5. Append JSONL
6. Update SQLite index
7. Append embeddings .npy

Renvoie un `IngestionResult` avec succès/échec et détails.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import ValidationError

from muses.schemas.row import Row
from muses.tables.embeddings import (
    DEFAULT_DIM,
    EmbeddingsCache,
    Encoder,
    StubEncoder,
)
from muses.tables.jsonl_io import append_row
from muses.tables.sqlite_index import create_schema, upsert_row


@dataclass
class IngestionResult:
    """Résultat d'une tentative d'ingestion."""

    success: bool
    row_id: str | None = None
    errors: list[str] = field(default_factory=list)
    stage_failed: str | None = None


@dataclass
class TablePaths:
    """Chemins d'une table : JSONL + SQLite index + npy embeddings."""

    jsonl: Path
    db: Path
    npy: Path

    @classmethod
    def from_dir(cls, table_dir: Path, table_name: str = "table") -> "TablePaths":
        d = Path(table_dir)
        return cls(
            jsonl=d / f"{table_name}.jsonl",
            db=d / f"{table_name}.sqlite",
            npy=d / f"{table_name}.embeddings.npy",
        )


def ingest(
    row_data: dict[str, Any],
    table_paths: TablePaths,
    *,
    encoder: Encoder | None = None,
    verify_signature: bool = True,
) -> IngestionResult:
    """Ingère une row dans la table indiquée.

    `row_data` est un dict (typiquement issu d'une requête HTTP) ; il est
    validé contre le schéma Row.

    Pour le MVP M0 :
    - `verify_signature` est un stub : on vérifie juste qu'une signature est
      présente quand `source` la requiert. La vérification cryptographique
      réelle viendra en M2/T22.
    - L'anonymisation est **assumée déjà faite par le caller**. Une validation
      de motifs résiduels (regex noms propres etc.) sera ajoutée en M1/T11.
    """
    encoder = encoder or StubEncoder(dim=DEFAULT_DIM)

    # 1. + 2. validation schéma + tags
    try:
        row = Row.model_validate(row_data)
    except ValidationError as exc:
        return IngestionResult(
            success=False,
            errors=[str(exc)],
            stage_failed="schema_validation",
        )

    # 3. signature (stub)
    if verify_signature:
        sig_errors = _check_signature_stub(row)
        if sig_errors:
            return IngestionResult(
                success=False,
                row_id=row.id,
                errors=sig_errors,
                stage_failed="signature",
            )

    # 4. anonymisation — assumée faite par le caller au M0 (cf. docstring module)

    # 5. append JSONL
    try:
        append_row(table_paths.jsonl, row)
    except OSError as exc:
        return IngestionResult(
            success=False,
            row_id=row.id,
            errors=[f"JSONL append failed: {exc}"],
            stage_failed="jsonl_append",
        )

    # 6. update SQLite index
    try:
        create_schema(table_paths.db)
        upsert_row(table_paths.db, row, table_paths.jsonl)
    except Exception as exc:  # SQLite et autres
        return IngestionResult(
            success=False,
            row_id=row.id,
            errors=[f"SQLite index update failed: {exc}"],
            stage_failed="sqlite_index",
        )

    # 7. append embeddings
    try:
        text = row.embeddable_text()
        embedding = encoder.encode([text])  # (1, dim)
        EmbeddingsCache(table_paths.npy).append(embedding)
    except Exception as exc:
        return IngestionResult(
            success=False,
            row_id=row.id,
            errors=[f"Embedding update failed: {exc}"],
            stage_failed="embeddings",
        )

    return IngestionResult(success=True, row_id=row.id)


def _check_signature_stub(row: Row) -> list[str]:
    """Vérification minimale pour M0 : présence de signature quand requise.

    Sources bootstrap et mined sont dispensées (cf. data-format.md §Schéma).
    Pour contribution_explicit et derived_from_edit, on vérifie juste que la
    signature n'est pas vide — la vérif cryptographique vient en M2/T22.
    """
    if row.source in ("bootstrap", "mined"):
        return []
    if not row.signature:
        return [f"source={row.source!r} requiert une signature non vide"]
    return []
