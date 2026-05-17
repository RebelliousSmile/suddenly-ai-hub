"""Tests T34 — Anti-sleeper guard."""

from datetime import datetime, timedelta, timezone

from muses.feedback.events import EventLog, FeedbackSignal
from muses.feedback.guardrails import AntiSleeperGuard


def _sig(signal, contributor, when):
    return FeedbackSignal(
        signal=signal,
        user_id="https://me.tld/users/alice",
        instance_id="me.tld",
        feature="dialogue",
        row_id="row-1",
        contributor_user_id=contributor,
        timestamp=when,
    )


def test_under_cap_not_flagged():
    guard = AntiSleeperGuard(daily_alpha_cap=10.0)
    now = datetime.now(tz=timezone.utc)
    signals = [_sig("accept", "https://bob.tld/users/bob", now) for _ in range(5)]
    result = guard.check_user("https://bob.tld/users/bob", signals)
    assert not result.over_threshold


def test_over_cap_flagged():
    guard = AntiSleeperGuard(daily_alpha_cap=10.0)
    now = datetime.now(tz=timezone.utc)
    signals = [_sig("accept", "https://bob.tld/users/bob", now) for _ in range(20)]
    result = guard.check_user("https://bob.tld/users/bob", signals)
    assert result.over_threshold
    assert result.daily_alpha_gain > 10.0


def test_old_signals_excluded():
    guard = AntiSleeperGuard(daily_alpha_cap=10.0)
    old = datetime.now(tz=timezone.utc) - timedelta(days=2)
    signals = [_sig("accept", "https://bob.tld/users/bob", old) for _ in range(20)]
    result = guard.check_user("https://bob.tld/users/bob", signals)
    assert result.daily_alpha_gain == 0.0


def test_scan_event_log(tmp_path):
    log = EventLog(tmp_path / "log.jsonl")
    now = datetime.now(tz=timezone.utc)
    for _ in range(15):
        log.append(_sig("accept", "https://attacker.tld/users/x", now))
    log.append(_sig("accept", "https://normal.tld/users/y", now))

    guard = AntiSleeperGuard(daily_alpha_cap=10.0)
    results = guard.scan_event_log(log)
    over = [r for r in results if r.over_threshold]
    assert any(r.user_id == "https://attacker.tld/users/x" for r in over)
    assert all(r.user_id != "https://normal.tld/users/y" for r in over)
