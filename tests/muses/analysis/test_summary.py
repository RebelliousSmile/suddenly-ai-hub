"""Tests T49 — Résumé de session."""

from muses.analysis.summary import generate_session_summary


def test_empty_session():
    s = generate_session_summary([])
    assert "vide" in s.lower()


def test_summary_mentions_scene_count():
    scenes = [
        ["Il dégaine son épée puis frappe.", "L'attaque le surprend."],
        ["Plus tard, ils discutent doucement avec une caresse."],
    ]
    s = generate_session_summary(scenes)
    assert "2 scène" in s


def test_summary_lists_detected_beats():
    scenes = [["Il dégaine son épée."]]  # hostilite
    s = generate_session_summary(scenes)
    assert "hostilite" in s
