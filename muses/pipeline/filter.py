"""Étage 4 — Filtreur.

v0 du MVP M2 : no-op passthrough (renvoie les top-N candidats du
recombinateur sans filtrage ni reranking). La version cross-encoder
appris sur signaux user vient en M5/T51 (cf. M3/T29 pour l'online learning
qui l'alimentera).
"""

from __future__ import annotations

from muses.pipeline.recombiner import RecombinedCandidate


class NoOpFilter:
    """Identity filter — sert tel quel les candidats du recombinateur."""

    def filter(
        self,
        candidates: list[RecombinedCandidate],
        *,
        top_n: int = 3,
    ) -> list[RecombinedCandidate]:
        return candidates[:top_n]
