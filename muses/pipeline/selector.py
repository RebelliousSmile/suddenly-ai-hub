"""Étage 1 — Sélecteur.

Filtre les tables disponibles selon les tags axiaux du contexte. Version v0 :
matching strict + fallback hiérarchique (relâche un axe à la fois dans
l'ordre `emotion_dominante → voix → rapport_initial → situation → univers`).

Voir architecture-tables-ml.md § Étage 1 et § Carte de couverture pour le
fallback. Version v2 (classifieur multi-label appris) en M5/T51.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from muses.schemas.tags import AXIS_NAMES, AxialTags
from muses.tables.jsonl_io import count_rows, iter_rows


# Ordre de relâchement : du plus local au plus fondamental
RELAX_ORDER: tuple[str, ...] = (
    "emotion_dominante",
    "voix",
    "rapport_initial",
    "situation",
    "univers",
)


@dataclass
class TableSelection:
    """Une table sélectionnée + les axes relâchés pour y arriver."""

    table_path: Path
    relaxed_axes: list[str]  # axes qu'on a dû ignorer du contexte

    @property
    def is_exact_match(self) -> bool:
        return not self.relaxed_axes


class TagMatchingSelector:
    """v0 : parcourt un set de tables candidates, filtre par compatibilité tags.

    Le `min_rows` est un seuil de peuplement : une table sélectionnée doit
    contenir au moins `min_rows` rows utiles. Si aucune table ne passe avec
    le contexte exact, on relâche un axe et on retente, jusqu'à épuisement
    de RELAX_ORDER.
    """

    def __init__(
        self,
        tables: list[Path],
        *,
        min_rows: int = 1,
    ):
        self.tables = [Path(p) for p in tables]
        self.min_rows = min_rows

    def select(self, context: AxialTags) -> list[TableSelection]:
        """Renvoie les tables compatibles. Fallback hiérarchique si vide."""
        ctx_dict = {axis: list(getattr(context, axis)) for axis in AXIS_NAMES}

        for n_relaxed in range(len(RELAX_ORDER) + 1):
            relaxed = list(RELAX_ORDER[:n_relaxed])
            relaxed_ctx = {
                axis: ([] if axis in relaxed else values)
                for axis, values in ctx_dict.items()
            }
            selections = self._select_with_context(relaxed_ctx, relaxed)
            if selections:
                return selections
        return []

    def _select_with_context(
        self,
        ctx_dict: dict[str, list[str]],
        relaxed: list[str],
    ) -> list[TableSelection]:
        selections: list[TableSelection] = []
        for table_path in self.tables:
            if not table_path.exists():
                continue
            if count_rows(table_path) < self.min_rows:
                continue
            if self._table_has_compatible_row(table_path, ctx_dict):
                selections.append(TableSelection(
                    table_path=table_path,
                    relaxed_axes=list(relaxed),
                ))
        return selections

    @staticmethod
    def _table_has_compatible_row(table_path: Path, ctx_dict: dict[str, list[str]]) -> bool:
        """Une table est sélectionnable s'il existe au moins une row compatible."""
        ctx_tags = AxialTags.model_construct(**ctx_dict)
        for row in iter_rows(table_path):
            if row.archived_at is not None:
                continue
            if row.tags.is_compatible_with(ctx_tags):
                return True
        return False
