"""Tests T36 — Meta-suggestions."""

from muses.feedback.meta_suggestions import MetaSuggestionGenerator
from muses.feedback.style_profile import StyleProfileStore


def test_no_observations_yields_no_suggestions(tmp_path):
    store = StyleProfileStore(tmp_path / "p.sqlite")
    gen = MetaSuggestionGenerator(store)
    assert gen.for_user("alice") == []


def test_overuse_detected(tmp_path):
    store = StyleProfileStore(tmp_path / "p.sqlite")
    # Concentre 80% des observations sur "hesitation"
    for _ in range(16):
        store.observe("alice", beat_label="hesitation")
    for _ in range(4):
        store.observe("alice", beat_label="provocation")
    gen = MetaSuggestionGenerator(store, overuse_threshold=0.5, min_observations=10)
    suggestions = gen.for_user("alice")
    families = {s.family for s in suggestions}
    labels = {s.label for s in suggestions}
    assert "overuse" in families
    assert "beat:hesitation" in labels


def test_below_min_observations_skipped(tmp_path):
    store = StyleProfileStore(tmp_path / "p.sqlite")
    store.observe("alice", beat_label="hesitation")
    gen = MetaSuggestionGenerator(store, min_observations=10)
    assert gen.for_user("alice") == []
