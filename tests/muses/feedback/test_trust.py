"""Tests T27 — Trust contextuel par auteur (Beta reputation)."""

from datetime import datetime, timedelta, timezone

import pytest

from muses.feedback.events import FeedbackSignal
from muses.feedback.trust import BetaReputation, TrustStore


def _signal(signal, contributor="https://other.tld/users/bob", tags=None):
    return FeedbackSignal(
        signal=signal,
        user_id="https://me.tld/users/alice",
        instance_id="me.tld",
        feature="dialogue",
        row_id="row-1",
        contributor_user_id=contributor,
        contributor_instance_id="other.tld",
        context_tags=tags or {"univers": ["medieval_fantastique"], "situation": ["combat"]},
    )


class TestBetaReputation:
    def test_mean_uniform_prior(self):
        rep = BetaReputation(alpha=1.0, beta=1.0, last_update=datetime.now(tz=timezone.utc))
        assert rep.mean() == 0.5

    def test_mean_strong_positive(self):
        rep = BetaReputation(alpha=99, beta=1, last_update=datetime.now(tz=timezone.utc))
        assert rep.mean() > 0.95

    def test_confidence_low_with_few_signals(self):
        rep = BetaReputation(alpha=2, beta=1, last_update=datetime.now(tz=timezone.utc))
        c = rep.confidence(prior_strength=10.0)
        assert c < 0.5

    def test_penalized_score_tends_to_neutral_with_low_confidence(self):
        rep = BetaReputation(alpha=4.5, beta=0.5, last_update=datetime.now(tz=timezone.utc))
        # Score brut = 0.9, mais peu d'observations → pénalisé
        score = rep.penalized_score(prior_strength=10.0)
        assert 0.5 < score < 0.9

    def test_decay_reduces_alpha_and_beta(self):
        past = datetime.now(tz=timezone.utc) - timedelta(days=180)
        rep = BetaReputation(alpha=10, beta=10, last_update=past)
        decayed = rep.decayed(datetime.now(tz=timezone.utc), half_life_days=180)
        # Demi-vie atteinte → alpha et beta divisés par ~2
        assert 4.5 < decayed.alpha < 5.5
        assert 4.5 < decayed.beta < 5.5


class TestTrustStore:
    def test_default_prior_when_unknown(self, tmp_path):
        store = TrustStore(tmp_path / "trust.sqlite")
        rep = store.get("unknown", "univers", "cyberpunk")
        assert rep.alpha == 1.0 and rep.beta == 1.0

    def test_update_increments_alpha_on_accept(self, tmp_path):
        store = TrustStore(tmp_path / "trust.sqlite")
        store.update_from_signal(_signal("accept"))
        rep = store.get("https://other.tld/users/bob", "univers", "medieval_fantastique")
        assert rep.alpha > 1.0

    def test_update_increments_beta_on_reject_off(self, tmp_path):
        store = TrustStore(tmp_path / "trust.sqlite")
        store.update_from_signal(_signal("reject_off"))
        rep = store.get("https://other.tld/users/bob", "univers", "medieval_fantastique")
        assert rep.beta > 1.0

    def test_reject_challenge_appreciated_is_neutral(self, tmp_path):
        store = TrustStore(tmp_path / "trust.sqlite")
        store.update_from_signal(_signal("reject_challenge_appreciated"))
        rep = store.get("https://other.tld/users/bob", "univers", "medieval_fantastique")
        # Pas d'update → reste sur le prior
        assert rep.alpha == 1.0 and rep.beta == 1.0

    def test_bootstrap_contributor_skipped(self, tmp_path):
        store = TrustStore(tmp_path / "trust.sqlite")
        n = store.update_from_signal(_signal("accept", contributor=None))
        assert n == 0

    def test_penalized_score_reasonable(self, tmp_path):
        store = TrustStore(tmp_path / "trust.sqlite")
        # 5 accepts répartis sur 5 jours (pour échapper au cap quotidien)
        for i in range(5):
            sig = _signal("accept")
            sig.timestamp = datetime.now(tz=timezone.utc) - timedelta(days=i)
            store.update_from_signal(sig)
        score = store.penalized_score(
            "https://other.tld/users/bob", "univers", "medieval_fantastique",
        )
        assert score > 0.5  # tendance positive
