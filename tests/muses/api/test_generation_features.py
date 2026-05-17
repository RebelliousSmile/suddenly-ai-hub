"""Tests T41-T44 — Endpoints des autres features de génération."""

import pytest
from fastapi.testclient import TestClient

from muses.api.server import create_app
from muses.ingestion.pipeline import TablePaths, ingest
from muses.tables.embeddings import StubEncoder


@pytest.fixture
def app(tmp_path):
    paths = TablePaths.from_dir(tmp_path / "tbl")
    encoder = StubEncoder(dim=8)
    for txt in ["fragment a", "fragment b", "fragment c"]:
        ingest({
            "level": "fragment",
            "tags": {"univers": ["medieval_fantastique"]},
            "content": {"text": txt, "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)
    return create_app(tables=[paths.jsonl], encoder=encoder)


SIG_HEADER = {"Signature": 'keyId="x",signature="abc"'}


@pytest.mark.parametrize("feature", ["action", "description", "thought", "video_prompt"])
def test_generation_endpoint_serves_correct_feature(app, feature):
    client = TestClient(app)
    resp = client.post(
        f"/v1/suggest/{feature}",
        json={
            "feature": feature,
            "context_text": "x",
            "context_tags": {"univers": ["medieval_fantastique"]},
        },
        headers=SIG_HEADER,
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["suggestions"]


@pytest.mark.parametrize("feature", ["action", "description", "thought", "video_prompt"])
def test_generation_endpoint_rejects_mismatched_feature(app, feature):
    client = TestClient(app)
    resp = client.post(
        f"/v1/suggest/{feature}",
        json={
            "feature": "dialogue",  # ne matche pas l'endpoint
            "context_text": "x",
            "context_tags": {},
        },
        headers=SIG_HEADER,
    )
    assert resp.status_code == 400
