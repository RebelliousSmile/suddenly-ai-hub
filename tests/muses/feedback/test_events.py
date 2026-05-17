"""Tests T26 — event log."""

from datetime import datetime, timezone

from muses.feedback.events import SIGNAL_TYPES, EventLog, FeedbackSignal


def _signal(signal="accept", **kw):
    defaults = dict(
        signal=signal,
        user_id="https://exemple.tld/users/alice",
        instance_id="exemple.tld",
        feature="dialogue",
        row_id="row-123",
        contributor_user_id="https://other.tld/users/bob",
        contributor_instance_id="other.tld",
        context_tags={"univers": ["medieval_fantastique"]},
    )
    defaults.update(kw)
    return FeedbackSignal(**defaults)


def test_signal_types_match_canon():
    assert set(SIGNAL_TYPES) == {
        "accept", "accept_edited", "reject_off",
        "reject_challenge_appreciated", "ignore",
    }


def test_append_and_count(tmp_path):
    log = EventLog(tmp_path / "log.jsonl")
    assert log.count() == 0
    log.append(_signal())
    log.append(_signal(signal="reject_off"))
    assert log.count() == 2


def test_round_trip_preserves_fields(tmp_path):
    log = EventLog(tmp_path / "log.jsonl")
    original = _signal(edited_text="version éditée")
    log.append(original)
    read_back = list(log.iter_signals())
    assert len(read_back) == 1
    assert read_back[0].signal == original.signal
    assert read_back[0].row_id == original.row_id
    assert read_back[0].edited_text == "version éditée"


def test_missing_file_empty(tmp_path):
    log = EventLog(tmp_path / "absent.jsonl")
    assert log.count() == 0
    assert list(log.iter_signals()) == []


def test_timestamps_preserved(tmp_path):
    log = EventLog(tmp_path / "log.jsonl")
    when = datetime(2026, 5, 17, 12, 0, 0, tzinfo=timezone.utc)
    log.append(_signal(timestamp=when))
    read_back = list(log.iter_signals())[0]
    assert read_back.timestamp == when
