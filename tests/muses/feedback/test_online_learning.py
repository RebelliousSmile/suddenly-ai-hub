"""Tests T29-T31 — Online learning v0 par (context, row)."""

from muses.feedback.events import FeedbackSignal
from muses.feedback.online_learning import OnlineLearner, _context_key


def _sig(signal, row_id="row-1"):
    return FeedbackSignal(
        signal=signal,
        user_id="https://me.tld/users/alice",
        instance_id="me.tld",
        feature="dialogue",
        row_id=row_id,
        context_tags={"univers": ["medieval_fantastique"], "situation": ["combat"]},
    )


def test_context_key_is_stable():
    a = {"univers": ["medieval_fantastique"], "situation": ["combat"]}
    b = {"situation": ["combat"], "univers": ["medieval_fantastique"]}
    assert _context_key(a) == _context_key(b)


def test_accept_increases_score(tmp_path):
    learner = OnlineLearner(tmp_path / "l.sqlite")
    learner.update_from_signal(_sig("accept"))
    score = learner.get_score(
        {"univers": ["medieval_fantastique"], "situation": ["combat"]}, "row-1",
    )
    assert score == 1.0


def test_reject_off_decreases_score(tmp_path):
    learner = OnlineLearner(tmp_path / "l.sqlite")
    learner.update_from_signal(_sig("accept"))
    learner.update_from_signal(_sig("reject_off"))
    score = learner.get_score(
        {"univers": ["medieval_fantastique"], "situation": ["combat"]}, "row-1",
    )
    assert score == 0.0  # (1 - 1) / 2


def test_reject_challenge_appreciated_is_neutral(tmp_path):
    learner = OnlineLearner(tmp_path / "l.sqlite")
    delta = learner.update_from_signal(_sig("reject_challenge_appreciated"))
    assert delta == 0.0


def test_unknown_pair_returns_zero(tmp_path):
    learner = OnlineLearner(tmp_path / "l.sqlite")
    assert learner.get_score({"univers": ["cyberpunk"]}, "no-such-row") == 0.0
