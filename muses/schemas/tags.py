"""Axes canoniques et validation des tags.

Source de vérité : aidd_docs/memory/external/axes-and-tags.md.
Toute évolution des sets canoniques doit passer par la procédure §7 du doc.
"""

from __future__ import annotations

from typing import ClassVar

from pydantic import BaseModel, Field, field_validator


AXIS_NAMES: tuple[str, ...] = (
    "univers",
    "situation",
    "rapport_initial",
    "voix",
    "emotion_dominante",
)


CANONICAL_VALUES: dict[str, frozenset[str]] = {
    "univers": frozenset({
        "medieval_fantastique",
        "historique_fantastique",
        "science_fiction",
        "space_opera",
        "cyberpunk",
        "steampunk",
        "post_apocalyptique",
        "contemporain_fantastique",
        "contemporain",
        "horreur_gothique",
        "super_heros",
        "oriental",
        "merveilleux",
        "humoristique",
        "univers_paralleles",
        "generique",
    }),
    "situation": frozenset({
        "combat",
        "romance",
        "intrigue",
        "politique",
        "enquete",
        "exploration",
        "introspection",
        "quotidien",
    }),
    "rapport_initial": frozenset({
        "hostile",
        "neutre",
        "amical",
    }),
    "voix": frozenset({
        "solennel",
        "narquois",
        "theatral",
        "neutre",
        "lyrique",
    }),
    "emotion_dominante": frozenset({
        "colere",
        "degout",
        "peur",
        "joie",
        "tristesse",
        "surprise",
    }),
}


class InvalidTagValue(ValueError):
    """Une valeur de tag n'appartient pas au set canonique de son axe."""


class AxialTags(BaseModel):
    """Tags d'une row sur les cinq axes canoniques.

    Chaque clé porte une liste de valeurs — une row peut être valide dans
    plusieurs valeurs d'un même axe. Liste vide = universel sur cet axe.
    """

    univers: list[str] = Field(default_factory=list)
    situation: list[str] = Field(default_factory=list)
    rapport_initial: list[str] = Field(default_factory=list)
    voix: list[str] = Field(default_factory=list)
    emotion_dominante: list[str] = Field(default_factory=list)

    AXES: ClassVar[tuple[str, ...]] = AXIS_NAMES

    @field_validator("univers", "situation", "rapport_initial", "voix", "emotion_dominante")
    @classmethod
    def _validate_axis_values(cls, values: list[str], info) -> list[str]:
        axis = info.field_name
        canonical = CANONICAL_VALUES[axis]
        invalid = [v for v in values if v not in canonical]
        if invalid:
            raise InvalidTagValue(
                f"Valeurs hors set canonique pour l'axe {axis!r}: {invalid}. "
                f"Set canonique : {sorted(canonical)}"
            )
        return values

    def is_compatible_with(self, context: "AxialTags") -> bool:
        """Une row est compatible avec un contexte si, sur chaque axe :
        - la row n'a pas de tag sur cet axe (universelle), ou
        - l'intersection entre les valeurs de la row et celles du contexte est non vide.
        """
        for axis in self.AXES:
            row_values = set(getattr(self, axis))
            if not row_values:
                continue
            ctx_values = set(getattr(context, axis))
            if not ctx_values:
                continue
            if row_values.isdisjoint(ctx_values):
                return False
        return True
