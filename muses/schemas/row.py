"""Schéma Pydantic de la row — l'unité élémentaire des tables Muses.

Voir aidd_docs/memory/external/data-format.md § Schéma commun à toutes les rows.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Annotated, Literal
from uuid import uuid4

from pydantic import BaseModel, Field, model_validator

from muses.schemas.content import (
    BeatContent,
    EntityContent,
    FragmentContent,
    TemplateContent,
)
from muses.schemas.tags import AxialTags


RowLevel = Literal["entity", "template", "beat", "fragment"]
RowSource = Literal["bootstrap", "contribution_explicit", "derived_from_edit", "mined"]


# Mapping level → classe content attendue. Source de la validation discriminée.
_CONTENT_TYPE_BY_LEVEL: dict[str, type[BaseModel]] = {
    "entity": EntityContent,
    "template": TemplateContent,
    "beat": BeatContent,
    "fragment": FragmentContent,
}


class Row(BaseModel):
    """Une row d'une table Muses, schéma générique sur les 4 niveaux."""

    id: str = Field(default_factory=lambda: str(uuid4()))
    level: RowLevel
    tags: AxialTags
    content: dict  # validé contre _CONTENT_TYPE_BY_LEVEL[level] dans _check_content_matches_level

    user_id: str | None = Field(
        None,
        description="Acteur ActivityPub (URI). null pour source bootstrap ou mined.",
    )
    instance_id: str | None = Field(
        None,
        description="Domaine de l'instance source. null pour source bootstrap.",
    )
    created_at: datetime = Field(default_factory=lambda: datetime.now(tz=timezone.utc))
    source: RowSource
    signature: str | None = Field(
        None,
        description="Signature HTTP ActivityPub (sauf bootstrap, mined).",
    )
    quality_score: float | None = Field(
        None,
        description="Score quality_gating, mis à jour en continu. Absent à l'ingestion.",
    )
    archived_at: datetime | None = Field(
        None,
        description="Date d'archivage si row sortie du service actif.",
    )

    @model_validator(mode="after")
    def _check_content_matches_level(self) -> "Row":
        """Le `content` doit être parsable comme le type attendu pour le `level`."""
        expected_type = _CONTENT_TYPE_BY_LEVEL[self.level]
        try:
            # Tentative de parsing : lève si invalide
            expected_type.model_validate(self.content)
        except Exception as exc:
            raise ValueError(
                f"Le champ content ne correspond pas au schéma {expected_type.__name__} "
                f"attendu pour level={self.level!r}: {exc}"
            ) from exc
        return self

    @model_validator(mode="after")
    def _check_provenance_consistency(self) -> "Row":
        """Cohérence entre source et présence des champs de provenance."""
        if self.source in ("bootstrap", "mined"):
            # user_id, signature peuvent être null ; instance_id peut être null pour bootstrap
            return self
        # source = contribution_explicit ou derived_from_edit : provenance complète requise
        missing = []
        if self.user_id is None:
            missing.append("user_id")
        if self.instance_id is None:
            missing.append("instance_id")
        if self.signature is None:
            missing.append("signature")
        if missing:
            raise ValueError(
                f"source={self.source!r} requiert les champs {missing} non-null"
            )
        return self

    def parsed_content(self) -> BaseModel:
        """Renvoie le content parsé dans son type concret."""
        return _CONTENT_TYPE_BY_LEVEL[self.level].model_validate(self.content)

    def embeddable_text(self) -> str:
        """Texte canonique extrait du content pour indexation FTS et embedding.

        Utilisé par l'index SQLite (sqlite_index.upsert_row) et le cache
        d'embeddings (embeddings.rebuild_for_table). Centralisé ici pour
        garantir que les deux indexent exactement le même texte.
        """
        parsed = self.parsed_content()
        if self.level == "fragment":
            return parsed.text
        if self.level == "entity":
            forms = list(parsed.forms.values()) if parsed.forms else []
            return " ".join([parsed.lemma] + forms)
        if self.level == "template":
            return parsed.skeleton
        if self.level == "beat":
            return f"{parsed.label} {parsed.description}"
        return ""
