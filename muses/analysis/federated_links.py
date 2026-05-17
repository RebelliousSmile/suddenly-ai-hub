"""T50 — Suggestions de liens fédérés.

Compare des PNJ d'une session aux personnages publics d'une instance par
similarité d'embeddings. Renvoie les top-K matches au-dessus d'un seuil,
classés en bandes de pertinence forte/moyenne/faible.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from muses.analysis.matcher import EmbeddingMatcher
from muses.tables.embeddings import Encoder


Confidence = Literal["fort", "moyen", "faible"]


@dataclass
class FederatedLinkSuggestion:
    """Une suggestion de lien entre PNJ session et personnage public."""

    session_character: str
    public_character_id: str
    similarity: float
    confidence: Confidence


def _classify_confidence(score: float) -> Confidence:
    if score >= 0.75:
        return "fort"
    if score >= 0.5:
        return "moyen"
    return "faible"


def find_federated_links(
    session_characters: dict[str, str],
    public_characters: dict[str, str],
    *,
    encoder: Encoder,
    threshold: float = 0.3,
    top_k_per_character: int = 3,
) -> list[FederatedLinkSuggestion]:
    """Pour chaque PNJ session, propose les K personnages publics les plus proches.

    `session_characters` : `{nom_session: description_textuelle}`.
    `public_characters` : `{id_public: description_textuelle}`.
    Encode les deux ensembles, calcule la similarité, classe en bandes.
    """
    if not session_characters or not public_characters:
        return []

    public_ids = list(public_characters.keys())
    public_descs = list(public_characters.values())
    public_embeddings = encoder.encode(public_descs)
    patterns = [
        (pub_id, {"description": desc}, public_embeddings[i])
        for i, (pub_id, desc) in enumerate(zip(public_ids, public_descs))
    ]
    matcher = EmbeddingMatcher(encoder, patterns)

    suggestions: list[FederatedLinkSuggestion] = []
    for session_name, session_desc in session_characters.items():
        matches = matcher.match(session_desc, top_k=top_k_per_character, min_score=threshold)
        for m in matches:
            suggestions.append(FederatedLinkSuggestion(
                session_character=session_name,
                public_character_id=m.pattern_label,
                similarity=m.score,
                confidence=_classify_confidence(m.score),
            ))
    return suggestions
