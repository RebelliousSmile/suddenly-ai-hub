"""Tests POST /v1/feedback/signal — capture des signaux + dispatch online."""

from fastapi.testclient import TestClient

from muses.api.server import create_app
from muses.feedback.events import EventLog
from muses.feedback.style_profile import StyleProfileStore
from muses.feedback.trust import TrustStore
from muses.ingestion.pipeline import TablePaths, ingest
from muses.tables.embeddings import StubEncoder


def _setup_app(tmp_path, with_feedback=True):
    paths = TablePaths.from_dir(tmp_path / "tbl")
    encoder = StubEncoder(dim=8)
    ingest({
        "level": "fragment",
        "tags": {"univers": ["medieval_fantastique"]},
        "content": {"text": "hello", "char_pov": "neutral"},
        "source": "bootstrap",
    }, paths, encoder=encoder, verify_signature=False)

    if with_feedback:
        app = create_app(
            tables=[paths.jsonl],
            encoder=encoder,
            event_log_path=tmp_path / "events.jsonl",
            trust_db_path=tmp_path / "trust.sqlite",
            style_db_path=tmp_path / "style.sqlite",
            learner_db_path=tmp_path / "learner.sqlite",
        )
    else:
        app = create_app(tables=[paths.jsonl], encoder=encoder)
    return app, tmp_path


def _payload(signal="accept"):
    return {
        "signal": signal,
        "user_id": "https://me.tld/users/alice",
        "instance_id": "me.tld",
        "feature": "dialogue",
        "row_id": "row-1",
        "contributor_user_id": "https://other.tld/users/bob",
        "contributor_instance_id": "other.tld",
        "context_tags": {"univers": ["medieval_fantastique"]},
    }


def test_signal_records_to_event_log(tmp_path):
    app, _ = _setup_app(tmp_path)
    client = TestClient(app)
    resp = client.post(
        "/v1/feedback/signal",
        json=_payload(),
        headers={"Signature": 'keyId="x",signature="abc"'},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["recorded"] is True
    log = EventLog(tmp_path / "events.jsonl")
    assert log.count() == 1


def test_signal_requires_auth(tmp_path):
    app, _ = _setup_app(tmp_path)
    client = TestClient(app)
    resp = client.post("/v1/feedback/signal", json=_payload())
    assert resp.status_code == 401


def test_signal_503_when_feedback_disabled(tmp_path):
    app, _ = _setup_app(tmp_path, with_feedback=False)
    client = TestClient(app)
    resp = client.post(
        "/v1/feedback/signal",
        json=_payload(),
        headers={"Signature": 'keyId="x",signature="abc"'},
    )
    assert resp.status_code == 503


def test_signal_dispatches_to_trust_store(tmp_path):
    app, base = _setup_app(tmp_path)
    client = TestClient(app)
    client.post(
        "/v1/feedback/signal",
        json=_payload("accept"),
        headers={"Signature": 'keyId="x",signature="abc"'},
    )
    trust = TrustStore(base / "trust.sqlite")
    rep = trust.get("https://other.tld/users/bob", "univers", "medieval_fantastique")
    assert rep.alpha > 1.0


def test_accept_dispatches_to_style_profile(tmp_path):
    app, base = _setup_app(tmp_path)
    client = TestClient(app)
    payload = _payload("accept_edited")
    payload["edited_text"] = "version finale du dialogue"
    client.post(
        "/v1/feedback/signal",
        json=payload,
        headers={"Signature": 'keyId="x",signature="abc"'},
    )
    style = StyleProfileStore(base / "style.sqlite")
    top_lex = {k for k, _ in style.top("https://me.tld/users/alice", "lex", limit=10)}
    assert "version" in top_lex or "dialogue" in top_lex
