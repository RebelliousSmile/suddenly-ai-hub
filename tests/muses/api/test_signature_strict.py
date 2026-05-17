"""Tests B5 — Vérification cryptographique des signatures HTTP."""

import base64
from email.utils import format_datetime
from datetime import datetime, timedelta, timezone

import pytest
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import FastAPI, Depends
from fastapi.testclient import TestClient

from muses.api.signature import (
    HttpKeyResolver,
    KeyResolver,
    SignatureInvalid,
    make_strict_dependency,
    verify_signature_strict,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------


def _generate_keypair() -> tuple[rsa.RSAPrivateKey, str]:
    priv = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    pub_pem = priv.public_key().public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode()
    return priv, pub_pem


def _build_signing_string(method: str, path: str, headers: dict[str, str], covered: list[str]) -> str:
    lines = []
    for h in covered:
        if h == "(request-target)":
            lines.append(f"(request-target): {method.lower()} {path}")
        else:
            lines.append(f"{h}: {headers[h]}")
    return "\n".join(lines)


def _sign(priv: rsa.RSAPrivateKey, signing_string: str) -> str:
    sig = priv.sign(
        signing_string.encode("utf-8"),
        padding.PKCS1v15(),
        hashes.SHA256(),
    )
    return base64.b64encode(sig).decode()


def _build_signature_header(*, key_id: str, headers: list[str], signature_b64: str) -> str:
    headers_str = " ".join(headers)
    return (
        f'keyId="{key_id}",algorithm="rsa-sha256",'
        f'headers="{headers_str}",signature="{signature_b64}"'
    )


class _StaticResolver:
    """KeyResolver injectable pour tests : mapping keyId → PEM."""

    def __init__(self, mapping: dict[str, str]):
        self._mapping = mapping

    def resolve(self, key_id: str) -> str:
        if key_id not in self._mapping:
            raise SignatureInvalid(f"keyId inconnu: {key_id}")
        return self._mapping[key_id]


def _make_test_app(resolver: KeyResolver, max_age_seconds: int = 300) -> FastAPI:
    app = FastAPI()
    dep = make_strict_dependency(resolver, max_age_seconds=max_age_seconds)

    @app.post("/protected")
    def protected(sig=Depends(dep)) -> dict:
        return {"key_id": sig.key_id}

    return app


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_valid_signature_accepted():
    priv, pub_pem = _generate_keypair()
    key_id = "https://exemple.tld/users/alice#main-key"
    date = format_datetime(datetime.now(tz=timezone.utc), usegmt=True)

    headers = {"host": "testserver", "date": date}
    covered = ["(request-target)", "host", "date"]
    sig_b64 = _sign(priv, _build_signing_string("POST", "/protected", headers, covered))
    sig_header = _build_signature_header(key_id=key_id, headers=covered, signature_b64=sig_b64)

    app = _make_test_app(_StaticResolver({key_id: pub_pem}))
    client = TestClient(app)
    resp = client.post(
        "/protected",
        json={},
        headers={**headers, "Signature": sig_header},
    )
    assert resp.status_code == 200, resp.text
    assert resp.json()["key_id"] == key_id


def test_tampered_body_does_not_break_signature_when_body_not_signed():
    """Si Digest n'est pas dans `headers`, modifier le body ne casse rien.

    C'est une limitation connue de draft-cavage : il faut inclure Digest
    pour protéger le body. Ce test documente le comportement.
    """
    priv, pub_pem = _generate_keypair()
    key_id = "https://exemple.tld/key"
    date = format_datetime(datetime.now(tz=timezone.utc), usegmt=True)

    headers = {"host": "testserver", "date": date}
    covered = ["(request-target)", "host", "date"]
    sig_b64 = _sign(priv, _build_signing_string("POST", "/protected", headers, covered))
    sig_header = _build_signature_header(key_id=key_id, headers=covered, signature_b64=sig_b64)

    app = _make_test_app(_StaticResolver({key_id: pub_pem}))
    client = TestClient(app)
    resp = client.post(
        "/protected",
        json={"tampered": "anything"},
        headers={**headers, "Signature": sig_header},
    )
    assert resp.status_code == 200


def test_wrong_signature_rejected():
    priv_attacker, _ = _generate_keypair()
    _, pub_legit = _generate_keypair()
    key_id = "https://exemple.tld/key"
    date = format_datetime(datetime.now(tz=timezone.utc), usegmt=True)

    headers = {"host": "testserver", "date": date}
    covered = ["(request-target)", "host", "date"]
    sig_b64 = _sign(priv_attacker, _build_signing_string("POST", "/protected", headers, covered))
    sig_header = _build_signature_header(key_id=key_id, headers=covered, signature_b64=sig_b64)

    app = _make_test_app(_StaticResolver({key_id: pub_legit}))
    client = TestClient(app)
    resp = client.post(
        "/protected",
        json={},
        headers={**headers, "Signature": sig_header},
    )
    assert resp.status_code == 401
    assert "invalide" in resp.json()["detail"].lower()


def test_replay_old_request_rejected():
    priv, pub_pem = _generate_keypair()
    key_id = "https://exemple.tld/key"
    # Date trop ancienne
    old = datetime.now(tz=timezone.utc) - timedelta(hours=1)
    date = format_datetime(old, usegmt=True)

    headers = {"host": "testserver", "date": date}
    covered = ["(request-target)", "host", "date"]
    sig_b64 = _sign(priv, _build_signing_string("POST", "/protected", headers, covered))
    sig_header = _build_signature_header(key_id=key_id, headers=covered, signature_b64=sig_b64)

    app = _make_test_app(_StaticResolver({key_id: pub_pem}), max_age_seconds=300)
    client = TestClient(app)
    resp = client.post(
        "/protected",
        json={},
        headers={**headers, "Signature": sig_header},
    )
    assert resp.status_code == 401
    assert "anti-replay" in resp.json()["detail"].lower()


def test_missing_date_rejected():
    priv, pub_pem = _generate_keypair()
    key_id = "https://exemple.tld/key"

    headers = {"host": "testserver"}
    covered = ["(request-target)", "host"]
    sig_b64 = _sign(priv, _build_signing_string("POST", "/protected", headers, covered))
    sig_header = _build_signature_header(key_id=key_id, headers=covered, signature_b64=sig_b64)

    app = _make_test_app(_StaticResolver({key_id: pub_pem}))
    client = TestClient(app)
    resp = client.post("/protected", json={}, headers={**headers, "Signature": sig_header})
    assert resp.status_code == 401
    assert "date" in resp.json()["detail"].lower()


def test_missing_signature_rejected():
    priv, pub_pem = _generate_keypair()
    app = _make_test_app(_StaticResolver({"x": pub_pem}))
    client = TestClient(app)
    resp = client.post("/protected", json={})
    assert resp.status_code == 401


def test_unknown_keyid_rejected():
    priv, pub_pem = _generate_keypair()
    date = format_datetime(datetime.now(tz=timezone.utc), usegmt=True)
    headers = {"host": "testserver", "date": date}
    covered = ["(request-target)", "host", "date"]
    sig_b64 = _sign(priv, _build_signing_string("POST", "/protected", headers, covered))
    sig_header = _build_signature_header(
        key_id="https://unknown.tld/key", headers=covered, signature_b64=sig_b64,
    )

    app = _make_test_app(_StaticResolver({"https://other.tld/key": pub_pem}))
    client = TestClient(app)
    resp = client.post("/protected", json={}, headers={**headers, "Signature": sig_header})
    assert resp.status_code == 401
    assert "inconnu" in resp.json()["detail"].lower()


def test_unsupported_algorithm_rejected():
    priv, pub_pem = _generate_keypair()
    key_id = "https://exemple.tld/key"
    date = format_datetime(datetime.now(tz=timezone.utc), usegmt=True)
    headers = {"host": "testserver", "date": date}
    covered = ["(request-target)", "host", "date"]
    sig_b64 = _sign(priv, _build_signing_string("POST", "/protected", headers, covered))
    sig_header = (
        f'keyId="{key_id}",algorithm="ed25519",headers="{" ".join(covered)}",'
        f'signature="{sig_b64}"'
    )

    app = _make_test_app(_StaticResolver({key_id: pub_pem}))
    client = TestClient(app)
    resp = client.post("/protected", json={}, headers={**headers, "Signature": sig_header})
    assert resp.status_code == 401
    assert "ed25519" in resp.json()["detail"]
