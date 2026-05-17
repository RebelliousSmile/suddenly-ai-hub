"""Étage 3 — Recombinateur.

Strictement déterministe : aucun modèle ML (cf. DECISIONS D04 et
architecture-tables-ml.md § Étage 3). Pour la feature MVP `suggest_dialogue`,
le recombinateur en v0 est un passthrough : il prend les N premières rows
de niveau `fragment` du Pondérateur et les sert telles quelles comme
candidats.

Les versions plus riches du recombinateur (assemblage beat → template →
entités) viendront avec les features `suggest_action`/`description`/`thought`
en M5/T41-T43.
"""

from __future__ import annotations

from dataclasses import dataclass

from muses.pipeline.weighter import WeightedRow


@dataclass
class RecombinedCandidate:
    """Candidat textuel produit par l'étage 3."""

    text: str
    source_row_ids: list[str]  # rows d'origine, pour la traçabilité
    source_scores: list[float]


class FragmentPassthroughRecombiner:
    """Pour la feature dialogue : sert les fragments tels quels.

    Filtre les rows non-fragments (silencieusement). Prend les `n` premiers
    et les renvoie. Le traçage de l'origine est conservé pour l'audit
    (cf. philosophy.md §6 Lisible et traçable).
    """

    def recombine(
        self,
        weighted: list[WeightedRow],
        *,
        n_candidates: int = 5,
    ) -> list[RecombinedCandidate]:
        candidates: list[RecombinedCandidate] = []
        for w in weighted:
            if w.row.level != "fragment":
                continue
            text = w.row.parsed_content().text
            candidates.append(RecombinedCandidate(
                text=text,
                source_row_ids=[w.row.id],
                source_scores=[w.score],
            ))
            if len(candidates) >= n_candidates:
                break
        return candidates
