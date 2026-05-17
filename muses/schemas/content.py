"""Schémas Pydantic du champ `content` par niveau de granularité.

Voir aidd_docs/memory/external/data-format.md § Champ content selon le niveau.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Entity
# ---------------------------------------------------------------------------


class EntityContent(BaseModel):
    """Unité lexicale typée (geste, émotion, lieu, objet, nom propre, trait…)."""

    type: str = Field(..., description="Type d'entité, ex: geste, emotion, lieu, objet")
    lemma: str = Field(..., description="Forme canonique de l'entité")
    variants: dict[str, list[str]] = Field(
        default_factory=dict,
        description="Dimensions de variation et leurs valeurs (genre, nombre, tense…)",
    )
    forms: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping clé combinatoire (ex: 'm.s.present') → forme effective",
    )


# ---------------------------------------------------------------------------
# Template
# ---------------------------------------------------------------------------


class TemplateSlot(BaseModel):
    """Spec d'un slot dans un template."""

    source: Literal["context", "static", "table:entities", "table:templates", "table:beats", "table:fragments"]
    type: str | None = Field(None, description="Type attendu si source=table:entities")
    value: str | None = Field(None, description="Valeur fixe si source=static")
    tags_match: bool = Field(False, description="Hériter des tags de la row template courante")


class TemplateContent(BaseModel):
    """Squelette de phrase avec slots typés."""

    skeleton: str = Field(..., description="Phrase avec slots {nom_slot} entre accolades")
    slots: dict[str, TemplateSlot] = Field(
        default_factory=dict,
        description="Mapping nom_slot → spec",
    )


# ---------------------------------------------------------------------------
# Beat
# ---------------------------------------------------------------------------


class BeatContent(BaseModel):
    """Unité narrative de niveau scène."""

    label: str = Field(..., description="Nom court du beat, ex: hesitation, revelation")
    description: str = Field(..., description="Description courte du beat")
    typical_templates: list[str] = Field(
        default_factory=list,
        description="IDs de templates typiquement utilisés pour ce beat",
    )
    arc_position: list[Literal["debut", "milieu", "tournant", "fin"]] = Field(
        default_factory=list,
        description="Positions typiques dans l'arc de scène ; vide = libre",
    )


# ---------------------------------------------------------------------------
# Fragment
# ---------------------------------------------------------------------------


class FragmentContent(BaseModel):
    """Sortie complète prête à insérer."""

    text: str = Field(..., description="Texte du fragment, prêt à insertion")
    char_pov: Literal["neutral", "pov_player", "third_person"] = Field(
        "neutral",
        description="POV du fragment",
    )
    beat_played: str | None = Field(
        None,
        description="Label du beat narratif incarné (optionnel, matching étage 3)",
    )
