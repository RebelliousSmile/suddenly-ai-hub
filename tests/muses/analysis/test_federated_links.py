"""Tests T50 — Suggestions de liens fédérés."""

from muses.analysis.federated_links import find_federated_links
from muses.tables.embeddings import StubEncoder


def test_empty_inputs_no_suggestions():
    encoder = StubEncoder(dim=16)
    assert find_federated_links({}, {"x": "y"}, encoder=encoder) == []
    assert find_federated_links({"a": "b"}, {}, encoder=encoder) == []


def test_finds_matches_above_threshold():
    encoder = StubEncoder(dim=16)
    suggestions = find_federated_links(
        session_characters={
            "Aldric": "Chevalier mélancolique poursuivant un secret familial",
        },
        public_characters={
            "char-123": "Chevalier mélancolique poursuivant un secret familial",
            "char-456": "Marchande cynique vendant des bibelots",
        },
        encoder=encoder,
        threshold=0.9,  # haut seuil → seulement le match exact passe
    )
    # Le stub encode pareillement les textes identiques → similarité = 1
    assert any(s.public_character_id == "char-123" for s in suggestions)
    assert all(s.public_character_id != "char-456" for s in suggestions)


def test_confidence_bands_assigned():
    encoder = StubEncoder(dim=16)
    suggestions = find_federated_links(
        session_characters={"Aldric": "même texte"},
        public_characters={"char-123": "même texte"},
        encoder=encoder,
        threshold=0.0,
    )
    assert suggestions
    assert suggestions[0].confidence == "fort"
