"""T36 — Méta-suggestions sur le style auteur (batch nocturne).

Voir style-coaching.md §4. v0 : trois familles de méta-suggestions générées
par observation simple du profil de style — pas de modèle ML, juste des
seuils et des règles.

Familles :
- "overuse" : un beat ou template surutilisé (>50% des observations récentes
  sur la dimension)
- "exploration" : axes contextuels sur lesquels l'auteur a peu écrit
- "anti_pattern" : asymétrie dans une métrique simple (à étendre)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from muses.feedback.style_profile import StyleProfileStore


MetaSuggestionFamily = Literal["overuse", "exploration", "anti_pattern"]


@dataclass
class MetaSuggestion:
    user_id: str
    family: MetaSuggestionFamily
    label: str       # ex: "beat:hesitation"
    score: float     # importance / confiance
    message: str     # texte affichable


class MetaSuggestionGenerator:
    """Génère des méta-suggestions à partir d'un StyleProfileStore."""

    def __init__(
        self,
        profile_store: StyleProfileStore,
        *,
        overuse_threshold: float = 0.5,
        min_observations: int = 10,
    ):
        self.profile_store = profile_store
        self.overuse_threshold = overuse_threshold
        self.min_observations = min_observations

    def for_user(self, user_id: str) -> list[MetaSuggestion]:
        """Calcule les méta-suggestions actuelles pour un user."""
        out: list[MetaSuggestion] = []
        out.extend(self._overuse(user_id, dimension="beat"))
        out.extend(self._overuse(user_id, dimension="template"))
        return out

    def _overuse(self, user_id: str, *, dimension: str) -> list[MetaSuggestion]:
        top = self.profile_store.top(user_id, dimension, limit=20)
        if not top:
            return []
        total = sum(count for _, count in top)
        if total < self.min_observations:
            return []
        out: list[MetaSuggestion] = []
        for key, count in top:
            share = count / total if total > 0 else 0
            if share >= self.overuse_threshold:
                out.append(MetaSuggestion(
                    user_id=user_id,
                    family="overuse",
                    label=f"{dimension}:{key}",
                    score=share,
                    message=(
                        f"Tu utilises {dimension} {key!r} dans {share:.0%} de tes "
                        f"observations récentes. Tester un autre {dimension} ?"
                    ),
                ))
        return out
