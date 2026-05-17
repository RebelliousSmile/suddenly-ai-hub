"""Tests T21 — serveur HTTP FastAPI."""

import pytest
from fastapi.testclient import TestClient

from muses.api.server import create_app
from muses.ingestion.pipeline import TablePaths, ingest
from muses.tables.embeddings import StubEncoder


@pytest.fixture
def app_with_table(tmp_path):
    """App FastAPI avec une table peuplée de 3 fragments."""
    paths = TablePaths.from_dir(tmp_path / "tbl")
    encoder = StubEncoder(dim=16)
    for text in ["fragment alpha", "fragment beta", "fragment gamma"]:
        ingest({
            "level": "fragment",
            "tags": {
                "univers": ["medieval_fantastique"],
                "situation": ["combat"],
            },
            "content": {"text": text, "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)

    return create_app(tables=[paths.jsonl], encoder=encoder)


def test_health_no_auth(app_with_table):
    client = TestClient(app_with_table)
    resp = client.get("/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert data["tables_count"] == 1


def test_suggest_dialogue_requires_signature(app_with_table):
    client = TestClient(app_with_table)
    resp = client.post("/v1/suggest/dialogue", json={
        "feature": "dialogue",
        "context_text": "alpha context",
        "context_tags": {"univers": ["medieval_fantastique"], "situation": ["combat"]},
    })
    assert resp.status_code == 401


def test_suggest_dialogue_returns_suggestions(app_with_table):
    client = TestClient(app_with_table)
    resp = client.post(
        "/v1/suggest/dialogue",
        json={
            "feature": "dialogue",
            "context_text": "alpha",
            "context_tags": {
                "univers": ["medieval_fantastique"],
                "situation": ["combat"],
            },
            "n_candidates": 3,
            "top_n": 2,
        },
        headers={"Signature": 'keyId="x",signature="abc"'},
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert len(data["suggestions"]) == 2
    assert all("text" in s for s in data["suggestions"])
    assert all(s["source_row_ids"] for s in data["suggestions"])
    assert data["selected_table_count"] == 1


def test_suggest_wrong_feature_rejected(app_with_table):
    client = TestClient(app_with_table)
    resp = client.post(
        "/v1/suggest/dialogue",
        json={
            "feature": "action",  # wrong endpoint
            "context_text": "x",
            "context_tags": {},
        },
        headers={"Signature": 'keyId="x",signature="abc"'},
    )
    assert resp.status_code == 400


def test_suggest_invalid_tags_rejected(app_with_table):
    client = TestClient(app_with_table)
    resp = client.post(
        "/v1/suggest/dialogue",
        json={
            "feature": "dialogue",
            "context_text": "x",
            "context_tags": {"univers": ["medieval-fantastique"]},  # hyphen invalide
        },
        headers={"Signature": 'keyId="x",signature="abc"'},
    )
    assert resp.status_code == 422  # Pydantic validation error
