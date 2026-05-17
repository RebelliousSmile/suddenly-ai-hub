"""T20 — Orchestrateur du pipeline 4 étages.

Chaîne sélecteur → pondérateur → recombinateur → filtreur. Renvoie un
`GenerationResult` qui inclut les candidats finaux et la traçabilité
intermédiaire (rows sélectionnées, scores, axes relâchés).

Voir architecture-tables-ml.md § Pipeline de génération.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from typing import Literal

from muses.pipeline.filter import NoOpFilter
from muses.pipeline.recombiner import FragmentPassthroughRecombiner, RecombinedCandidate
from muses.pipeline.selector import TableSelection, TagMatchingSelector
from muses.pipeline.weighter import CosineWeighter, WeightedRow
from muses.schemas.tags import AXIS_NAMES, AxialTags
from muses.tables.embeddings import Encoder


Mode = Literal["confort", "challenge"]


# Multiplicateur appliqué aux rows familières en mode challenge (cf.
# style-coaching.md §2 — "malus borné dans [0.3, 1.0]"). 0.5 = malus médian.
CHALLENGE_FAMILIAR_MULTIPLIER = 0.5


def _build_relaxed_tags(original: AxialTags, relaxed_axes: list[str]) -> AxialTags:
    """Construit un AxialTags où les axes relâchés sont vidés."""
    data = {axis: list(getattr(original, axis)) for axis in AXIS_NAMES}
    for axis in relaxed_axes:
        data[axis] = []
    return AxialTags.model_construct(**data)


@dataclass
class Candidate:
    """Candidat final retourné par l'orchestrateur."""

    text: str
    source_row_ids: list[str]
    source_scores: list[float]


@dataclass
class GenerationResult:
    """Résultat complet d'une génération, candidats + traçabilité."""

    candidates: list[Candidate]
    selected_tables: list[TableSelection] = field(default_factory=list)
    relaxed_axes: list[str] = field(default_factory=list)
    weighted_count: int = 0


class Orchestrator:
    """Coordonne les 4 étages pour produire des candidats à partir d'un contexte."""

    def __init__(
        self,
        tables: list[Path],
        encoder: Encoder,
        *,
        selector: TagMatchingSelector | None = None,
        weighter: CosineWeighter | None = None,
        recombiner: FragmentPassthroughRecombiner | None = None,
        filter_stage: NoOpFilter | None = None,
        style_store=None,  # StyleProfileStore | None — utilisé en mode challenge
    ):
        self.tables = [Path(p) for p in tables]
        self.encoder = encoder
        self.selector = selector or TagMatchingSelector(self.tables)
        self.weighter = weighter or CosineWeighter(encoder)
        self.recombiner = recombiner or FragmentPassthroughRecombiner()
        self.filter_stage = filter_stage or NoOpFilter()
        self.style_store = style_store

    def generate(
        self,
        *,
        context_text: str,
        context_tags: AxialTags,
        n_candidates: int = 5,
        top_n: int = 3,
        mode: Mode = "confort",
        user_id: str | None = None,
    ) -> GenerationResult:
        """Pipeline complet : sélectionne, pondère, recombine, filtre.

        En mode "challenge" et si `user_id` et `style_store` sont fournis,
        les rows familières au user (dans son top-20 d'observations) reçoivent
        un malus multiplicatif après le pondérateur — réduction de pertinence
        sans exclusion (cf. style-coaching.md §2).
        """
        # Étage 1 — Sélecteur
        selections = self.selector.select(context_tags)
        if not selections:
            return GenerationResult(candidates=[])

        selected_paths = [s.table_path for s in selections]
        relaxed_axes = selections[0].relaxed_axes if selections else []
        # Le weighter doit appliquer les MÊMES axes relâchés que le sélecteur :
        # sinon une row sélectionnée parce qu'on a relâché un axe se ferait
        # filtrer ici. Cf. tests/muses/pipeline/test_orchestrator.py
        # ::test_orchestrator_no_matching_tables_returns_empty.
        relaxed_tags = _build_relaxed_tags(context_tags, relaxed_axes)

        # Étage 2 — Pondérateur
        weighted = self.weighter.rank(
            selected_paths,
            context_text,
            context_tags=relaxed_tags,
        )

        # Mode challenge : malus sur les rows familières du user (cf. T33)
        if mode == "challenge" and user_id and self.style_store is not None:
            weighted = self._apply_challenge_malus(weighted, user_id)

        # Étage 3 — Recombinateur
        recombined = self.recombiner.recombine(weighted, n_candidates=n_candidates)

        # Étage 4 — Filtreur
        filtered = self.filter_stage.filter(recombined, top_n=top_n)

        return GenerationResult(
            candidates=[
                Candidate(
                    text=c.text,
                    source_row_ids=c.source_row_ids,
                    source_scores=c.source_scores,
                )
                for c in filtered
            ],
            selected_tables=selections,
            relaxed_axes=relaxed_axes,
            weighted_count=len(weighted),
        )

    def _apply_challenge_malus(
        self,
        weighted: list[WeightedRow],
        user_id: str,
    ) -> list[WeightedRow]:
        """Réduit le score des rows familières et re-trie."""
        familiar_ids = {
            key for key, _ in self.style_store.top(user_id, "row", limit=20)
        }
        if not familiar_ids:
            return weighted
        adjusted = []
        for w in weighted:
            mult = CHALLENGE_FAMILIAR_MULTIPLIER if w.row.id in familiar_ids else 1.0
            adjusted.append(WeightedRow(
                row=w.row,
                score=w.score * mult,
                table_path=w.table_path,
            ))
        adjusted.sort(key=lambda x: x.score, reverse=True)
        return adjusted
