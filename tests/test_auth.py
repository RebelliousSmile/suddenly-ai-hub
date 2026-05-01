from __future__ import annotations

import base64
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi.testclient import TestClient

from gateway.auth import (
    build_signing_string,
    clear_key_cache,
    parse_signature_header,
    preload_key,
    verify_rsa_sha256,
)
from gateway.main import app


# ---------------------------------------------------------------------------
# Module-scope RSA key pair
# ---------------------------------------------------------------------------

_PRIVATE_KEY = rsa.generate_private_key(public_exponent=65537, key_size=2048)
_PUBLIC_KEY = _PRIVATE_KEY.public_key()
_KEY_ID = "https://rp.example.com/actor#main-key"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sign(signing_string: str) -> str:
    sig = _PRIVATE_KEY.sign(signing_string.encode(), padding.PKCS1v15(), hashes.SHA256())
    return base64.b64encode(sig).decode()


def _make_sig_header(headers_list: list[str], signing_string: str) -> str:
    sig_b64 = _sign(signing_string)
    headers_str = " ".join(headers_list)
    return f'keyId="{_KEY_ID}",algorithm="rsa-sha256",headers="{headers_str}",signature="{sig_b64}"'


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def _clear_cache():
    clear_key_cache()
    yield
    clear_key_cache()


@pytest.fixture
def client():
    preload_key(_KEY_ID, _PUBLIC_KEY)
    with TestClient(app) as c:
        yield c


# ---------------------------------------------------------------------------
# parse_signature_header
# ---------------------------------------------------------------------------

class TestParseSignatureHeader:
    def test_valid_header(self):
        hdr = 'keyId="https://example.com/actor#key",algorithm="rsa-sha256",headers="(request-target) date",signature="abc123"'
        params = parse_signature_header(hdr)
        assert params["keyId"] == "https://example.com/actor#key"
        assert params["algorithm"] == "rsa-sha256"
        assert params["headers"] == "(request-target) date"
        assert params["signature"] == "abc123"

    def test_missing_key_id_raises(self):
        hdr = 'algorithm="rsa-sha256",headers="date",signature="abc"'
        with pytest.raises(ValueError, match="keyId"):
            parse_signature_header(hdr)

    def test_missing_headers_raises(self):
        hdr = 'keyId="https://example.com/actor#key",signature="abc"'
        with pytest.raises(ValueError, match="headers"):
            parse_signature_header(hdr)

    def test_missing_signature_raises(self):
        hdr = 'keyId="https://example.com/actor#key",headers="date"'
        with pytest.raises(ValueError, match="signature"):
            parse_signature_header(hdr)


# ---------------------------------------------------------------------------
# build_signing_string
# ---------------------------------------------------------------------------

class TestBuildSigningString:
    def test_request_target(self):
        result = build_signing_string(["(request-target)"], {}, "POST", "/v1/chat/completions")
        assert result == "(request-target): post /v1/chat/completions"

    def test_regular_header(self):
        result = build_signing_string(["date"], {"date": "Thu, 01 May 2026 12:00:00 GMT"}, "POST", "/v1/chat/completions")
        assert result == "date: Thu, 01 May 2026 12:00:00 GMT"

    def test_missing_header_raises(self):
        with pytest.raises(ValueError, match="date"):
            build_signing_string(["date"], {}, "POST", "/v1/chat/completions")

    def test_multiple_headers_joined_by_newline(self):
        result = build_signing_string(
            ["(request-target)", "date"],
            {"date": "Thu, 01 May 2026 12:00:00 GMT"},
            "POST",
            "/v1/chat/completions",
        )
        lines = result.split("\n")
        assert len(lines) == 2
        assert lines[0] == "(request-target): post /v1/chat/completions"
        assert lines[1] == "date: Thu, 01 May 2026 12:00:00 GMT"


# ---------------------------------------------------------------------------
# verify_rsa_sha256
# ---------------------------------------------------------------------------

class TestVerifyRsaSha256:
    def test_valid_signature_passes(self):
        msg = "test signing string"
        sig_b64 = _sign(msg)
        verify_rsa_sha256(_PUBLIC_KEY, msg, sig_b64)  # no exception

    def test_invalid_signature_raises(self):
        msg = "test signing string"
        bad_sig = base64.b64encode(b"invalide" * 32).decode()
        with pytest.raises(InvalidSignature):
            verify_rsa_sha256(_PUBLIC_KEY, msg, bad_sig)

    def test_wrong_key_raises(self):
        msg = "test signing string"
        other_private = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        sig_b64 = base64.b64encode(
            other_private.sign(msg.encode(), padding.PKCS1v15(), hashes.SHA256())
        ).decode()
        with pytest.raises(InvalidSignature):
            verify_rsa_sha256(_PUBLIC_KEY, msg, sig_b64)


# ---------------------------------------------------------------------------
# verify_http_signature (intégration via TestClient)
# ---------------------------------------------------------------------------

class TestVerifyHttpSignatureDependency:
    def _make_request(self, client, signing_string: str, headers_list: list[str], extra_headers: dict | None = None) -> object:
        sig_header = _make_sig_header(headers_list, signing_string)
        headers = {"signature": sig_header, "content-type": "application/json"}
        if extra_headers:
            headers.update(extra_headers)
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock) as mock_chat:
            from gateway.models import ChatChoice, ChatResponse, ChatUsage, Message
            mock_chat.return_value = ChatResponse(
                id="cmpl-test",
                object="chat.completion",
                model="suddenly-7b",
                adapter_used=None,
                choices=[ChatChoice(index=0, message=Message(role="assistant", content="ok"), finish_reason="stop")],
                usage=ChatUsage(prompt_tokens=5, completion_tokens=5, total_tokens=10),
            )
            return client.post(
                "/v1/chat/completions",
                json={"model": "suddenly-7b", "messages": [{"role": "user", "content": "action"}]},
                headers=headers,
            )

    def test_missing_signature_header_returns_401(self, client):
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "suddenly-7b", "messages": [{"role": "user", "content": "action"}]},
            )
        assert resp.status_code == 401

    def test_malformed_signature_header_returns_400(self, client):
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "suddenly-7b", "messages": [{"role": "user", "content": "action"}]},
                headers={"signature": "not=valid,garbage"},
            )
        assert resp.status_code == 400

    def test_valid_signature_returns_200(self, client):
        signing_string = "(request-target): post /v1/chat/completions"
        resp = self._make_request(client, signing_string, ["(request-target)"])
        assert resp.status_code == 200

    def test_invalid_signature_returns_401(self, client):
        bad_sig = base64.b64encode(b"invalide" * 32).decode()
        sig_header = f'keyId="{_KEY_ID}",algorithm="rsa-sha256",headers="(request-target)",signature="{bad_sig}"'
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "suddenly-7b", "messages": [{"role": "user", "content": "action"}]},
                headers={"signature": sig_header, "content-type": "application/json"},
            )
        assert resp.status_code == 401

    def test_missing_listed_header_returns_400(self, client):
        signing_string = "(request-target): post /v1/chat/completions"
        sig_header = _make_sig_header(["(request-target)", "date"], signing_string)
        with patch("gateway.main.vllm_client.chat", new_callable=AsyncMock):
            resp = client.post(
                "/v1/chat/completions",
                json={"model": "suddenly-7b", "messages": [{"role": "user", "content": "action"}]},
                headers={"signature": sig_header, "content-type": "application/json"},
                # note: no "date" header sent
            )
        assert resp.status_code == 400
