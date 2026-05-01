from __future__ import annotations

import base64
import re
from typing import Any

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding
from fastapi import HTTPException, Request

_key_cache: dict[str, Any] = {}

_SIG_PARAM_RE = re.compile(r'(\w+)="([^"]*)"')

_REQUIRED_FIELDS = {"keyId", "headers", "signature"}


def parse_signature_header(header: str) -> dict[str, str]:
    params = dict(_SIG_PARAM_RE.findall(header))
    missing = _REQUIRED_FIELDS - params.keys()
    if missing:
        raise ValueError(f"Missing Signature fields: {missing}")
    return params


async def fetch_public_key(key_id: str):
    if key_id in _key_cache:
        return _key_cache[key_id]
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.get(key_id, headers={"Accept": "application/activity+json"})
            resp.raise_for_status()
            data = resp.json()
    except (httpx.RequestError, httpx.HTTPStatusError) as exc:
        raise HTTPException(status_code=502, detail=f"Cannot fetch keyId: {exc}") from exc

    pem = (data.get("publicKey") or {}).get("publicKeyPem")
    if not pem:
        raise HTTPException(status_code=502, detail="Missing publicKey.publicKeyPem")

    try:
        public_key = serialization.load_pem_public_key(pem.encode())
    except Exception as exc:
        raise HTTPException(status_code=502, detail=f"Invalid PEM key: {exc}") from exc

    _key_cache[key_id] = public_key
    return public_key


def build_signing_string(headers_list: list[str], request_headers: dict, method: str, path: str) -> str:
    lines = []
    for name in headers_list:
        if name == "(request-target)":
            lines.append(f"(request-target): {method.lower()} {path}")
        else:
            value = request_headers.get(name.lower())
            if value is None:
                raise ValueError(f"Header listed in Signature but missing from request: {name!r}")
            lines.append(f"{name}: {value}")
    return "\n".join(lines)


def verify_rsa_sha256(public_key, signing_string: str, signature_b64: str) -> None:
    sig_bytes = base64.b64decode(signature_b64)
    public_key.verify(sig_bytes, signing_string.encode(), padding.PKCS1v15(), hashes.SHA256())


def preload_key(key_id: str, public_key) -> None:
    _key_cache[key_id] = public_key


def clear_key_cache() -> None:
    _key_cache.clear()


async def verify_http_signature(request: Request) -> None:
    header = request.headers.get("signature")
    if not header:
        raise HTTPException(status_code=401, detail="Missing Signature header")

    try:
        sig_params = parse_signature_header(header)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=f"Malformed Signature header: {exc}") from exc

    headers_list = sig_params["headers"].split()
    method = request.method
    path = request.url.path
    if request.url.query:
        path = f"{path}?{request.url.query}"

    request_headers = dict(request.headers)

    try:
        signing_string = build_signing_string(headers_list, request_headers, method, path)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    public_key = await fetch_public_key(sig_params["keyId"])

    try:
        verify_rsa_sha256(public_key, signing_string, sig_params["signature"])
    except InvalidSignature as exc:
        raise HTTPException(status_code=401, detail="Invalid HTTP signature") from exc
