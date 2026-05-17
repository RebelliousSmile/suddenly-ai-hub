"""Tests T35 — endpoint /v1/admin/coverage + module compute_coverage."""

from fastapi.testclient import TestClient

from muses.api.admin import compute_coverage
from muses.api.server import create_app
from muses.ingestion.pipeline import TablePaths, ingest
from muses.tables.embeddings import StubEncoder


def _ingest(table_dir, tags, n=2):
    paths = TablePaths.from_dir(table_dir)
    encoder = StubEncoder(dim=8)
    for i in range(n):
        ingest({
            "level": "fragment",
            "tags": tags,
            "content": {"text": f"text {i}", "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)
    return paths.jsonl


def test_compute_coverage_aggregates(tmp_path):
    table = _ingest(tmp_path / "tbl", {
        "univers": ["medieval_fantastique"], "situation": ["combat"],
    }, n=3)
    cells = compute_coverage([table])
    assert len(cells) == 1
    cell = cells[0]
    assert cell.counts_by_level["fragment"] == 3
    assert cell.last_contribution is not None


def test_compute_coverage_multiple_cells(tmp_path):
    t1 = _ingest(tmp_path / "med", {"univers": ["medieval_fantastique"]})
    t2 = _ingest(tmp_path / "cyber", {"univers": ["cyberpunk"]})
    cells = compute_coverage([t1, t2])
    assert len(cells) == 2


def test_admin_endpoint_no_token_when_admin_disabled(tmp_path):
    table = _ingest(tmp_path / "tbl", {"univers": ["medieval_fantastique"]})
    app = create_app(tables=[table], admin_token=None)
    client = TestClient(app)
    resp = client.get("/v1/admin/coverage")
    assert resp.status_code == 200
    data = resp.json()
    assert data["total_cells"] == 1


def test_admin_endpoint_requires_token_when_set(tmp_path):
    table = _ingest(tmp_path / "tbl", {"univers": ["medieval_fantastique"]})
    app = create_app(tables=[table], admin_token="s3cret")
    client = TestClient(app)
    resp = client.get("/v1/admin/coverage")
    assert resp.status_code == 403
    resp = client.get("/v1/admin/coverage", headers={"X-Admin-Token": "wrong"})
    assert resp.status_code == 403
    resp = client.get("/v1/admin/coverage", headers={"X-Admin-Token": "s3cret"})
    assert resp.status_code == 200
