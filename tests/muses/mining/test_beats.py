"""Tests du classifieur heuristique de beats."""

from muses.mining.beats import BEAT_KEYWORDS, classify_beat_keywords


def test_known_beats_listed():
    assert set(BEAT_KEYWORDS.keys()) >= {
        "hesitation", "provocation", "menace", "revelation",
        "hostilite", "tendresse",
    }


def test_detects_provocation():
    text = "Voilà tout ce que tu as ? lance-t-il en riant."
    beats = classify_beat_keywords(text)
    assert "provocation" in beats


def test_detects_menace():
    text = "« Tu vas le regretter », murmure-t-il."
    beats = classify_beat_keywords(text)
    assert "menace" in beats


def test_detects_hostilite():
    text = "Il sort son épée et frappe."
    beats = classify_beat_keywords(text)
    assert "hostilite" in beats


def test_multiple_beats_possible():
    text = "Il dégaine son épée puis hésite face à l'inconnu."
    beats = classify_beat_keywords(text)
    assert set(beats) >= {"hostilite", "hesitation"}


def test_no_match_returns_empty():
    text = "Le ciel est gris et la pluie tombe."
    beats = classify_beat_keywords(text)
    assert beats == []


def test_case_insensitive():
    text = "VOILÀ TOUT CE QUE TU AS"
    beats = classify_beat_keywords(text)
    assert "provocation" in beats
