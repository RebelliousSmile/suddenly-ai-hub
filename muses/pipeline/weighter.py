"""Étage 2 — Pondérateur.

v0 : similarité cosinus entre embedding du contexte et embeddings des rows
(pré-calculés et persistés en .npy adjacent au JSONL). Renvoie une liste
ordonnée par score décroissant.

Voir architecture-tables-ml.md § Étage 2. Version v2 (cross-encoder appris
sur signaux user) en M5/T51.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from muses.schemas.row import Row
from muses.schemas.tags import AxialTags
from muses.tables.embeddings import EmbeddingsCache, Encoder
from muses.tables.jsonl_io import read_rows


@dataclass
class WeightedRow:
    """Row associée à un score de pertinence calculé par l'étage 2."""

    row: Row
    score: float
    table_path: Path


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Similarité cosinus entre vecteur a (dim,) et matrice b (n, dim).

    Renvoie un vecteur de scores (n,). Si une norme est nulle, le score
    correspondant est 0.
    """
    a_norm = np.linalg.norm(a)
    b_norms = np.linalg.norm(b, axis=1)
    denom = a_norm * b_norms
    # Évite la division par zéro
    with np.errstate(invalid="ignore", divide="ignore"):
        scores = np.where(denom > 0, b @ a / denom, 0.0)
    return scores.astype(np.float32)


class CosineWeighter:
    """Pondère les rows d'une ou plusieurs tables par similarité cosinus."""

    def __init__(self, encoder: Encoder):
        self.encoder = encoder

    def rank(
        self,
        table_paths: list[Path],
        context_text: str,
        *,
        context_tags: AxialTags | None = None,
        top_k: int | None = None,
    ) -> list[WeightedRow]:
        """Renvoie les rows de toutes les tables, triées par similarité décroissante.

        `context_tags` filtre les rows compatibles avant scoring (en plus du
        filtrage déjà fait par l'étage 1). Si `None`, toutes les rows sont scorées.

        `top_k` limite la liste de sortie. None = pas de limite.
        """
        if not table_paths:
            return []

        ctx_embedding = self.encoder.encode([context_text])[0]

        all_weighted: list[WeightedRow] = []
        for table_path in table_paths:
            table_path = Path(table_path)
            npy_path = table_path.with_suffix(".embeddings.npy")
            if not npy_path.exists():
                # Pas d'embeddings cache : on skippe cette table
                continue
            rows = read_rows(table_path)
            embeddings = EmbeddingsCache(npy_path).load()
            if len(rows) != embeddings.shape[0]:
                raise RuntimeError(
                    f"Désynchronisation JSONL/embeddings pour {table_path}: "
                    f"{len(rows)} rows vs {embeddings.shape[0]} embeddings"
                )

            scores = _cosine_similarity(ctx_embedding, embeddings)
            for row, score in zip(rows, scores):
                if row.archived_at is not None:
                    continue
                if context_tags is not None and not row.tags.is_compatible_with(context_tags):
                    continue
                all_weighted.append(WeightedRow(
                    row=row,
                    score=float(score),
                    table_path=table_path,
                ))

        all_weighted.sort(key=lambda w: w.score, reverse=True)
        if top_k is not None:
            return all_weighted[:top_k]
        return all_weighted
