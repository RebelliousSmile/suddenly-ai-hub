"""Tests T47-T50 — Endpoints d'analyse."""

import pytest
from fastapi.testclient import TestClient

from muses.api.server import create_app
from muses.ingestion.pipeline import TablePaths, ingest
from muses.tables.embeddings import StubEncoder


@pytest.fixture
def app(tmp_path):
    paths = TablePaths.from_dir(tmp_path / "tbl")
    encoder = StubEncoder(dim=8)
    ingest({
        "level": "fragment",
        "tags": {"univers": ["medieval_fantastique"]},
        "content": {"text": "hello", "char_pov": "neutral"},
        "source": "bootstrap",
    }, paths, encoder=encoder, verify_signature=False)
    return create_app(tables=[paths.jsonl], encoder=encoder)


SIG_HEADER = {"Signature": 'keyId="x",signature="abc"'}


def test_consistency_scene(app):
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze/consistency_scene",
        json={"scene_fragments": [
            "Il dégaine son épée.",
            "Il caresse doucement sa main.",
        ]},
        headers=SIG_HEADER,
    )
    assert resp.status_code == 200, resp.text
    data = resp.json()
    assert data["n_issues"] == 1


def test_consistency_session(app):
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze/consistency_session",
        json={"scenes": [
            ["Il attaque.", "Il frappe."],
        ]},
        headers=SIG_HEADER,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["n_scenes"] == 1


def test_summary_endpoint(app):
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze/summary",
        json={"scenes": [["Il dégaine son épée puis frappe avec rage."]]},
        headers=SIG_HEADER,
    )
    assert resp.status_code == 200
    assert "Résumé" in resp.json()["summary"]


def test_federated_links_endpoint(app):
    client = TestClient(app)
    resp = client.post(
        "/v1/analyze/federated_links",
        json={
            "session_characters": {"Aldric": "Chevalier mélancolique"},
            "public_characters": {"char-x": "Chevalier mélancolique"},
            "threshold": 0.0,
        },
        headers=SIG_HEADER,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["suggestions"]
