"""Schémas Pydantic des requêtes et réponses HTTP du service Muses.

Pour la feature MVP `suggest_dialogue` (use-cases.md §2.1).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from muses.schemas.tags import AxialTags


class SuggestRequest(BaseModel):
    """Requête de suggestion. Contexte minimal pour les features de génération."""

    feature: str = Field(..., description="Nom de la feature, ex: dialogue, action, description, thought, video_prompt")
    context_text: str = Field(
        ...,
        description="Texte du contexte (derniers reports, fiche perso compilée, etc.)",
    )
    context_tags: AxialTags = Field(
        ...,
        description="Tags axiaux du contexte (5 axes canoniques)",
    )
    n_candidates: int = Field(5, ge=1, le=20)
    top_n: int = Field(3, ge=1, le=10)
    mode: str = Field("confort", pattern="^(confort|challenge)$")
    user_id: str | None = Field(
        None,
        description="ID auteur pour mode challenge (obligatoire si mode=challenge)",
    )


class SuggestionItem(BaseModel):
    """Une suggestion individuelle avec sa traçabilité."""

    text: str
    source_row_ids: list[str] = Field(default_factory=list)
    source_scores: list[float] = Field(default_factory=list)


class SuggestResponse(BaseModel):
    """Réponse complète. Inclut traçabilité globale (axes relâchés, etc.)."""

    suggestions: list[SuggestionItem]
    relaxed_axes: list[str] = Field(
        default_factory=list,
        description="Axes du contexte qu'il a fallu relâcher pour trouver des tables",
    )
    selected_table_count: int = 0
    weighted_count: int = 0
