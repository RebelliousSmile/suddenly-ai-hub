"""Tests des schémas Row + content selon le niveau."""

from datetime import datetime, timezone

import pytest
from pydantic import ValidationError

from muses.schemas.content import (
    BeatContent,
    EntityContent,
    FragmentContent,
    TemplateContent,
    TemplateSlot,
)
from muses.schemas.row import Row
from muses.schemas.tags import AxialTags


def _make_row(level: str, content: dict, source: str = "bootstrap", **kwargs):
    """Constructeur helper."""
    defaults = {
        "level": level,
        "tags": AxialTags(univers=["medieval_fantastique"]),
        "content": content,
        "source": source,
    }
    if source not in ("bootstrap", "mined"):
        defaults.update({
            "user_id": "https://exemple.tld/users/alice",
            "instance_id": "exemple.tld",
            "signature": "keyId=\"...\",sig=\"...\"",
        })
    defaults.update(kwargs)
    return Row(**defaults)


class TestContentSchemas:
    def test_entity_content(self):
        e = EntityContent(
            type="geste",
            lemma="serrer les poings",
            variants={"genre": ["m", "f"]},
            forms={"m.s.present": "serre les poings"},
        )
        assert e.type == "geste"

    def test_template_content(self):
        t = TemplateContent(
            skeleton="{char} {geste} en {emotion}",
            slots={
                "char": TemplateSlot(source="context"),
                "geste": TemplateSlot(source="table:entities", type="geste"),
                "emotion": TemplateSlot(source="table:entities", type="emotion"),
            },
        )
        assert "{char}" in t.skeleton

    def test_fragment_content_default_pov(self):
        f = FragmentContent(text="« Tu rigoles ? »")
        assert f.char_pov == "neutral"

    def test_fragment_invalid_pov_rejected(self):
        with pytest.raises(ValidationError):
            FragmentContent(text="x", char_pov="pov-player")  # hyphen — invalide

    def test_beat_arc_position_constrained(self):
        with pytest.raises(ValidationError):
            BeatContent(label="x", description="y", arc_position=["middle"])  # not in literal


class TestRow:
    def test_minimal_bootstrap_row(self):
        row = _make_row("fragment", {"text": "« Hello »"})
        assert row.level == "fragment"
        assert row.source == "bootstrap"
        assert row.user_id is None
        assert row.signature is None
        assert row.created_at <= datetime.now(tz=timezone.utc)

    def test_contribution_requires_provenance(self):
        with pytest.raises(ValidationError) as exc_info:
            Row(
                level="fragment",
                tags=AxialTags(),
                content={"text": "x"},
                source="contribution_explicit",
                # missing user_id, instance_id, signature
            )
        msg = str(exc_info.value)
        assert "user_id" in msg
        assert "signature" in msg

    def test_content_must_match_level(self):
        with pytest.raises(ValidationError):
            _make_row("entity", {"text": "x"})  # text est pour fragment, pas entity

    def test_parsed_content_returns_typed_object(self):
        row = _make_row("fragment", {"text": "« Hello »"})
        parsed = row.parsed_content()
        assert isinstance(parsed, FragmentContent)
        assert parsed.text == "« Hello »"

    def test_id_auto_generated(self):
        row1 = _make_row("fragment", {"text": "a"})
        row2 = _make_row("fragment", {"text": "b"})
        assert row1.id != row2.id

    def test_id_preserved_if_provided(self):
        row = _make_row("fragment", {"text": "x"}, id="custom-uuid")
        assert row.id == "custom-uuid"

    def test_invalid_source_rejected(self):
        with pytest.raises(ValidationError):
            _make_row("fragment", {"text": "x"}, source="unknown_source")

    def test_invalid_tag_value_rejected(self):
        with pytest.raises(ValidationError):
            Row(
                level="fragment",
                tags=AxialTags(univers=["medieval-fantastique"]),  # hyphen
                content={"text": "x"},
                source="bootstrap",
            )

    def test_round_trip_json(self):
        row = _make_row("fragment", {"text": "« Tu rigoles ? »"})
        as_json = row.model_dump_json()
        parsed = Row.model_validate_json(as_json)
        assert parsed.id == row.id
        assert parsed.content == row.content
