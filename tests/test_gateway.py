from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest
from fastapi.testclient import TestClient

from gateway.adapter_router import resolve_adapter
from gateway.config import GatewayConfig, set_config
from gateway.main import app
from gateway.models import ChatChoice, ChatResponse, ChatUsage, Message


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_config():
    cfg = GatewayConfig(
        available_adapters=frozenset(["lora-cyberpunk", "lora-combat", "lora-generique"]),
    )
    set_config(cfg)
    yield cfg
    set_config(GatewayConfig())


@pytest.fixture
def client():
    with TestClient(app) as c:
        yield c


def _fake_chat_response(adapter: str | None = None) -> ChatResponse:
    return ChatResponse(
        id="cmpl-test",
        object="chat.completion",
        model="suddenly-7b",
        adapter_used=adapter,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content="Réponse du MJ."),
                finish_reason="stop",
            )
        ],
        usage=ChatUsage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


def _valid_chat_payload(**overrides) -> dict:
    base = {
        "model": "suddenly-7b",
        "messages": [
            {"role": "user", "content": "Le joueur entre dans la taverne."}
        ],
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# adapter_router
# ---------------------------------------------------------------------------

class TestAdapterRouter:
    def test_both_genre_situation_merged_available(self):
        set_config(GatewayConfig(available_adapters=frozenset(["lora-cyberpunk-combat"])))
        assert resolve_adapter("cyberpunk", "combat") == "lora-cyberpunk-combat"

    def test_fallback_to_situation(self):
        set_config(GatewayConfig(available_adapters=frozenset(["lora-combat"])))
        assert resolve_adapter("cyberpunk", "combat") == "lora-combat"

    def test_fallback_to_genre(self):
        set_config(GatewayConfig(available_adapters=frozenset(["lora-cyberpunk"])))
        assert resolve_adapter("cyberpunk", "combat") == "lora-cyberpunk"

    def test_fallback_to_generique(self):
        set_config(GatewayConfig(available_adapters=frozenset(["lora-generique"])))
        assert resolve_adapter("cyberpunk", "combat") == "lora-generique"

    def test_fallback_to_base_model(self):
        set_config(GatewayConfig(available_adapters=frozenset()))
        assert resolve_adapter("cyberpunk", "combat") is None

    def test_no_genre_no_situation_generique(self):
        set_config(GatewayConfig(available_adapters=frozenset(["lora-generique"])))
        assert resolve_adapter(None, None) == "lora-generique"

    def test_no_genre_no_situation_base(self):
        set_config(GatewayConfig(available_adapters=frozenset()))
        assert resolve_adapter(None, None) is None

    def test_situation_only(self):
        set_config(GatewayConfig(available_adapters=frozenset(["lora-combat"])))
        assert resolve_adapter(None, "combat") == "lora-combat"

    def test_genre_only(self):
        set_config(GatewayConfig(available_adapters=frozenset(["lora-scifi"])))
        assert resolve_adapter("scifi", None) == "lora-scifi"


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_vllm_unreachable(self, client):
        mock_cls = MagicMock()
        mock_cls.return_value.__aenter__ = AsyncMock(side_effect=httpx.ConnectError("refused"))
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("gateway.main.httpx.AsyncClient", mock_cls):
            resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "degraded"
        assert data["vllm_reachable"] is False
        assert data["models_loaded"] == 0

    def test_vllm_reachable(self, client):
        mock_resp = MagicMock()
        mock_resp.is_success = True
        mock_inner = AsyncMock()
        mock_inner.get = AsyncMock(return_value=mock_resp)
        mock_cls = MagicMock()
        mock_cls.return_value.__aenter__ = AsyncMock(return_value=mock_inner)
        mock_cls.return_value.__aexit__ = AsyncMock(return_value=False)
        with patch("gateway.main.httpx.AsyncClient", mock_cls):
            resp = client.get("/v1/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert data["vllm_reachable"] is True
        assert data["models_loaded"] > 0


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class TestModels:
    def test_returns_configured_models(self, client):
        resp = client.get("/v1/models")
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "list"
        ids = [m["id"] for m in data["data"]]
        assert "suddenly-7b" in ids
        assert "suddenly-13b" in ids

    def test_adapters_listed(self, client):
        resp = client.get("/v1/models")
        for model in resp.json()["data"]:
            assert "lora-generique" in model["available_adapters"]


# ---------------------------------------------------------------------------
# GET /v1/stats
# ---------------------------------------------------------------------------

class TestStats:
    def test_returns_stats(self, client):
        resp = client.get("/v1/stats")
        assert resp.status_code == 200
        data = resp.json()
        assert data["models_available"] == 2
        assert data["adapters_active"] == 3
        assert data["sessions_contributed"] == 0


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

class TestChatCompletions:
    def test_valid_request(self, client):
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = _fake_chat_response()
            resp = client.post("/v1/chat/completions", json=_valid_chat_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["choices"][0]["message"]["role"] == "assistant"

    def test_unknown_genre_returns_422(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json=_valid_chat_payload(genre="fantasy-unknown"),
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "valid_values" in detail

    def test_unknown_situation_returns_422(self, client):
        resp = client.post(
            "/v1/chat/completions",
            json=_valid_chat_payload(situation="unknown-situation"),
        )
        assert resp.status_code == 422
        detail = resp.json()["detail"]
        assert "valid_values" in detail

    def test_valid_genre_and_situation(self, client):
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = _fake_chat_response("lora-combat")
            resp = client.post(
                "/v1/chat/completions",
                json=_valid_chat_payload(genre="cyberpunk", situation="combat"),
            )
        assert resp.status_code == 200
        assert resp.json()["adapter_used"] == "lora-combat"

    def test_vllm_http_error_returns_502(self, client):
        mock_resp = MagicMock()
        mock_resp.status_code = 500
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = httpx.HTTPStatusError("err", request=MagicMock(), response=mock_resp)
            resp = client.post("/v1/chat/completions", json=_valid_chat_payload())
        assert resp.status_code == 502

    def test_vllm_connect_error_returns_503(self, client):
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock) as mock_chat:
            mock_chat.side_effect = httpx.ConnectError("refused")
            resp = client.post("/v1/chat/completions", json=_valid_chat_payload())
        assert resp.status_code == 503

    def test_no_messages_returns_422(self, client):
        resp = client.post("/v1/chat/completions", json={"model": "suddenly-7b", "messages": []})
        assert resp.status_code == 422


# ---------------------------------------------------------------------------
# POST /v1/contribute
# ---------------------------------------------------------------------------

class TestContribute:
    def _valid_payload(self, **overrides) -> dict:
        base = {
            "messages": [
                {"role": "user", "content": "action"},
                {"role": "assistant", "content": "réponse MJ"},
            ],
            "source_instance": "https://rp.example.com",
        }
        base.update(overrides)
        return base

    def test_valid_contribution(self, client):
        resp = client.post("/v1/contribute", json=self._valid_payload())
        assert resp.status_code == 200
        data = resp.json()
        assert data["accepted"] is True
        assert "session_id" in data

    def test_each_session_gets_unique_id(self, client):
        ids = {client.post("/v1/contribute", json=self._valid_payload()).json()["session_id"] for _ in range(3)}
        assert len(ids) == 3

    def test_unknown_genre_returns_422(self, client):
        resp = client.post("/v1/contribute", json=self._valid_payload(genre="unknown"))
        assert resp.status_code == 422

    def test_unknown_situation_returns_422(self, client):
        resp = client.post("/v1/contribute", json=self._valid_payload(situation="unknown"))
        assert resp.status_code == 422

    def test_single_message_rejected(self, client):
        payload = self._valid_payload()
        payload["messages"] = [{"role": "user", "content": "seul"}]
        resp = client.post("/v1/contribute", json=payload)
        assert resp.status_code == 422
