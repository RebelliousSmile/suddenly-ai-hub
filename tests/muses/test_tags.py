"""Tests des axes canoniques et de la validation des tags."""

import pytest
from pydantic import ValidationError

from muses.schemas.tags import (
    AXIS_NAMES,
    CANONICAL_VALUES,
    AxialTags,
    InvalidTagValue,
)


class TestAxisCanon:
    def test_five_axes_exactly(self):
        assert len(AXIS_NAMES) == 5
        assert set(AXIS_NAMES) == {
            "univers",
            "situation",
            "rapport_initial",
            "voix",
            "emotion_dominante",
        }

    def test_canonical_values_match_axes(self):
        assert set(CANONICAL_VALUES.keys()) == set(AXIS_NAMES)

    def test_ekman_six_emotions(self):
        assert CANONICAL_VALUES["emotion_dominante"] == frozenset({
            "colere", "degout", "peur", "joie", "tristesse", "surprise",
        })

    def test_rapport_initial_three_values(self):
        assert CANONICAL_VALUES["rapport_initial"] == frozenset({
            "hostile", "neutre", "amical",
        })

    def test_no_hyphenated_identifiers(self):
        """Identifiants ASCII snake_case sans tiret (cf. axes-and-tags.md §Principes)."""
        for axis, values in CANONICAL_VALUES.items():
            assert "-" not in axis, f"axis name {axis!r} contains hyphen"
            for v in values:
                assert "-" not in v, f"value {v!r} on axis {axis!r} contains hyphen"


class TestAxialTagsValidation:
    def test_empty_tags_valid(self):
        tags = AxialTags()
        assert tags.univers == []

    def test_valid_value_accepted(self):
        tags = AxialTags(univers=["medieval_fantastique"], situation=["combat"])
        assert tags.univers == ["medieval_fantastique"]

    def test_invalid_value_rejected(self):
        with pytest.raises(ValidationError) as exc_info:
            AxialTags(univers=["medieval-fantastique"])  # hyphen
        assert "Valeurs hors set canonique" in str(exc_info.value)

    def test_obsolete_genre_value_rejected(self):
        """`genre` n'est plus une valeur d'axe (cf. DECISIONS D05)."""
        with pytest.raises(ValidationError):
            AxialTags(univers=["genre"])

    def test_polyvalent_row_supported(self):
        """Une row peut être valide sur plusieurs valeurs d'un même axe."""
        tags = AxialTags(rapport_initial=["hostile", "neutre"])
        assert len(tags.rapport_initial) == 2


class TestCompatibility:
    def test_universal_row_matches_anything(self):
        row = AxialTags()  # toutes listes vides = universelle
        ctx = AxialTags(univers=["cyberpunk"])
        assert row.is_compatible_with(ctx)

    def test_matching_intersection(self):
        row = AxialTags(univers=["medieval_fantastique", "horreur_gothique"])
        ctx = AxialTags(univers=["horreur_gothique"])
        assert row.is_compatible_with(ctx)

    def test_disjoint_blocks(self):
        row = AxialTags(univers=["cyberpunk"])
        ctx = AxialTags(univers=["medieval_fantastique"])
        assert not row.is_compatible_with(ctx)

    def test_axis_absent_from_context_is_neutral(self):
        """Si le contexte n'a pas de tag sur un axe, la row passe sur cet axe."""
        row = AxialTags(voix=["narquois"])
        ctx = AxialTags()  # pas de voix
        assert row.is_compatible_with(ctx)

    def test_all_five_axes_must_match(self):
        row = AxialTags(
            univers=["cyberpunk"],
            situation=["combat"],
            voix=["narquois"],
        )
        ctx_good = AxialTags(
            univers=["cyberpunk"],
            situation=["combat"],
            voix=["narquois"],
        )
        ctx_bad_voix = AxialTags(
            univers=["cyberpunk"],
            situation=["combat"],
            voix=["solennel"],
        )
        assert row.is_compatible_with(ctx_good)
        assert not row.is_compatible_with(ctx_bad_voix)
