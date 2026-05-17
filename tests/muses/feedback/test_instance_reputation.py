"""Tests T28 — Instance reputation + multiplicateur."""

from muses.feedback.instance_reputation import (
    MULTIPLIER_DEFAULT,
    MULTIPLIER_MAX,
    MULTIPLIER_MIN,
    InstanceReputationStore,
    clamp_multiplier,
)


def test_default_is_neutral(tmp_path):
    store = InstanceReputationStore(tmp_path / "ir.sqlite")
    rep = store.get("unknown.tld")
    assert rep.base_multiplier == MULTIPLIER_DEFAULT
    assert rep.source == "auto"


def test_admin_override_persists(tmp_path):
    store = InstanceReputationStore(tmp_path / "ir.sqlite")
    store.set_admin_override("trusted.tld", 1.5)  # sera clamped
    rep = store.get("trusted.tld")
    assert rep.base_multiplier == MULTIPLIER_MAX
    assert rep.source == "admin"


def test_clamp_bounds():
    assert clamp_multiplier(2.0) == MULTIPLIER_MAX
    assert clamp_multiplier(0.1) == MULTIPLIER_MIN
    assert clamp_multiplier(0.8) == 0.8


def test_auto_update_lowers_for_poor_quality(tmp_path):
    store = InstanceReputationStore(tmp_path / "ir.sqlite")
    store.update_auto("lax.tld", accept_rate=0.2, n_signals=200)
    rep = store.get("lax.tld")
    assert rep.base_multiplier < MULTIPLIER_DEFAULT
    assert rep.source == "auto"


def test_auto_update_smoothed_by_prior(tmp_path):
    store = InstanceReputationStore(tmp_path / "ir.sqlite")
    # Faible volume → forte smoothing par prior 0.6 → multiplicateur proche du neutre
    store.update_auto("new.tld", accept_rate=1.0, n_signals=1)
    rep = store.get("new.tld")
    assert rep.base_multiplier < MULTIPLIER_MAX  # pas saturé malgré 100% accept
