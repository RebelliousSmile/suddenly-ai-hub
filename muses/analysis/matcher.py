"""T45 — Embedder + Matcher + Agrégateur du pipeline d'analyse.

Pattern : projection du contenu utilisateur sur des tables de patterns
(au lieu de tirage depuis des tables de contenu). Cf. architecture-
tables-ml.md § Pipeline d'analyse — projection inversée.

`EmbeddingMatcher` : trouve les patterns les plus proches d'un texte donné
en cosinus sur embeddings préalablement calculés.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from muses.pipeline.weighter import _cosine_similarity
from muses.tables.embeddings import Encoder


@dataclass
class MatchResult:
    """Un pattern matché contre un texte d'entrée, avec son score."""

    pattern_label: str
    score: float
    pattern_payload: dict | None = None


class EmbeddingMatcher:
    """Projette un texte sur un set de patterns par similarité cosinus.

    Les patterns sont fournis sous forme de tuples (label, payload, embedding).
    Le matcher renvoie les top-K plus proches au-dessus d'un seuil.
    """

    def __init__(
        self,
        encoder: Encoder,
        patterns: list[tuple[str, dict, np.ndarray]],
    ):
        self.encoder = encoder
        # On stacke les embeddings une fois pour vectoriser le match
        self._labels = [p[0] for p in patterns]
        self._payloads = [p[1] for p in patterns]
        if patterns:
            self._matrix = np.stack([p[2] for p in patterns]).astype(np.float32)
        else:
            self._matrix = np.zeros((0, encoder.dim), dtype=np.float32)

    def match(
        self,
        text: str,
        *,
        top_k: int = 5,
        min_score: float = 0.0,
    ) -> list[MatchResult]:
        """Renvoie les top-K patterns avec un score >= min_score, par ordre décroissant."""
        if self._matrix.shape[0] == 0:
            return []
        query = self.encoder.encode([text])[0]
        scores = _cosine_similarity(query, self._matrix)
        ranked = sorted(
            zip(self._labels, self._payloads, scores),
            key=lambda x: x[2],
            reverse=True,
        )
        return [
            MatchResult(pattern_label=label, score=float(score), pattern_payload=payload)
            for label, payload, score in ranked
            if score >= min_score
        ][:top_k]
