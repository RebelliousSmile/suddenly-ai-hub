"""T25 — Smoke test end-to-end : MusesClient ↔ FastAPI app.

Démontre le flux complet d'une instance Suddenly mockée vers le service
Muses : préparation de la requête, signature stub, parsing de la réponse,
extraction des suggestions et de la traçabilité.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from muses.api.server import create_app
from muses.client import MusesClient
from muses.ingestion.pipeline import TablePaths, ingest
from muses.schemas.tags import AxialTags
from muses.tables.embeddings import StubEncoder


@pytest.fixture
def live_client(tmp_path):
    """Crée une app Muses peuplée + un MusesClient branché via TestClient.

    TestClient est une sous-classe `httpx.Client` configurée pour servir une
    app ASGI in-process — exactement ce qu'attend le MusesClient.
    """
    paths = TablePaths.from_dir(tmp_path / "tbl")
    encoder = StubEncoder(dim=16)
    for text in [
        "« Tu n'iras pas plus loin », gronde le chevalier.",
        "Le marchand lève son arme avec hésitation.",
        "Une lueur d'ironie traverse son regard.",
        "Il frappe du plat de la lame, sans rage apparente.",
    ]:
        ingest({
            "level": "fragment",
            "tags": {
                "univers": ["medieval_fantastique"],
                "situation": ["combat"],
                "rapport_initial": ["hostile"],
                "voix": ["solennel"],
                "emotion_dominante": ["colere"],
            },
            "content": {"text": text, "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)

    app = create_app(tables=[paths.jsonl], encoder=encoder)
    test_http_client = TestClient(app)
    client = MusesClient(http_client=test_http_client)
    yield client
    client.close()
    test_http_client.close()


def test_e2e_health_check(live_client):
    info = live_client.health()
    assert info["status"] == "ok"
    assert info["tables_count"] == 1


def test_e2e_suggest_dialogue(live_client):
    result = live_client.suggest(
        feature="dialogue",
        context_text="Le chevalier brandit son épée face au voleur.",
        context_tags=AxialTags(
            univers=["medieval_fantastique"],
            situation=["combat"],
            rapport_initial=["hostile"],
            voix=["solennel"],
            emotion_dominante=["colere"],
        ),
        signature='keyId="https://exemple.tld/users/alice#main-key",algorithm="rsa-sha256",signature="dGVzdA=="',
        n_candidates=3,
        top_n=2,
    )

    assert len(result.suggestions) == 2
    for sugg in result.suggestions:
        assert sugg.text
        assert sugg.source_row_ids
        assert sugg.source_scores
    assert result.selected_table_count == 1
    assert result.weighted_count >= 2


def test_e2e_unsigned_request_is_rejected(live_client):
    with pytest.raises(httpx.HTTPStatusError) as exc_info:
        live_client._client.post(  # bypass MusesClient.suggest pour tester l'auth
            "/v1/suggest/dialogue",
            json={
                "feature": "dialogue",
                "context_text": "x",
                "context_tags": {"univers": ["medieval_fantastique"]},
            },
        ).raise_for_status()
    assert exc_info.value.response.status_code == 401


def test_e2e_unsupported_feature_in_client():
    """Le client rejette les features non supportées avant même de hit le serveur."""
    client = MusesClient(base_url="http://nowhere")
    with pytest.raises(ValueError, match="non supportée"):
        client.suggest(
            feature="action",
            context_text="x",
            context_tags=AxialTags(),
            signature="x",
        )
    client.close()
