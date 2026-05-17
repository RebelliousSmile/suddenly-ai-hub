"""Tests T32 — Profil de style auteur."""

from muses.feedback.style_profile import StyleProfileStore


def test_observe_and_top(tmp_path):
    store = StyleProfileStore(tmp_path / "p.sqlite")
    store.observe("alice", beat_label="hesitation")
    store.observe("alice", beat_label="hesitation")
    store.observe("alice", beat_label="provocation")
    top = store.top("alice", "beat", limit=10)
    assert top[0][0] == "hesitation"
    assert top[0][1] > top[1][1]


def test_text_observation_indexes_tokens(tmp_path):
    store = StyleProfileStore(tmp_path / "p.sqlite")
    store.observe("alice", text="le chevalier brandit son épée fièrement")
    top = store.top("alice", "lex", limit=5)
    keys = {k for k, _ in top}
    assert "chevalier" in keys
    assert "épée" in keys


def test_purge_user_removes_all(tmp_path):
    store = StyleProfileStore(tmp_path / "p.sqlite")
    store.observe("alice", beat_label="x")
    store.observe("alice", template_id="t1")
    n = store.purge_user("alice")
    assert n == 2
    assert store.top("alice", "beat") == []


def test_minimum_observations_threshold(tmp_path):
    store = StyleProfileStore(tmp_path / "p.sqlite")
    assert not store.has_minimum_observations("alice", threshold=5)
    for i in range(10):
        store.observe("alice", row_id=f"row-{i}")
    assert store.has_minimum_observations("alice", threshold=5)
