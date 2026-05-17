"""Vérification cryptographique des signatures HTTP ActivityPub draft-cavage.

Mode strict (production) — implémente :
1. Parsing du header `Signature` (déjà fait par `auth.parse_http_signature`).
2. Reconstruction du signing string canonique depuis les headers listés.
3. Résolution de l'acteur (`GET keyId` avec `Accept: application/activity+json`).
4. Vérification RSA-SHA256 contre `publicKey.publicKeyPem` de l'acteur.
5. Anti-replay : rejet si `Date` plus ancien que `max_age_seconds`.

La résolution de l'acteur est cachée en mémoire (TTL paramétrable). Pour les
tests, on injecte un `KeyResolver` qui renvoie directement les clés sans
réseau.

Référence : draft-cavage-http-signatures-12 (encore utilisé par ActivityPub).
"""

from __future__ import annotations

import base64
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Callable, Protocol

import httpx
from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa
from fastapi import HTTPException, Request, status

from muses.api.auth import ParsedSignature, parse_http_signature


logger = logging.getLogger("muses.signature")


class SignatureInvalid(HTTPException):
    """Signature rejetée — toujours 401 vers le caller."""

    def __init__(self, detail: str):
        super().__init__(status_code=status.HTTP_401_UNAUTHORIZED, detail=detail)


class KeyResolver(Protocol):
    """Récupère la clé publique PEM pour un `keyId` donné."""

    def resolve(self, key_id: str) -> str:
        ...


class HttpKeyResolver:
    """Résolveur par défaut : fetch l'acteur ActivityPub par HTTP avec cache TTL.

    L'URL du keyId désigne typiquement `https://instance/users/x#main-key`.
    On fetch `https://instance/users/x` (le `#fragment` est ignoré côté HTTP)
    avec `Accept: application/activity+json` et on lit `publicKey.publicKeyPem`.
    """

    def __init__(
        self,
        *,
        ttl_seconds: int = 3600,
        timeout_seconds: float = 5.0,
        http_client: httpx.Client | None = None,
    ):
        self.ttl_seconds = ttl_seconds
        self._cache: dict[str, tuple[float, str]] = {}
        self._client = http_client or httpx.Client(timeout=timeout_seconds)

    def resolve(self, key_id: str) -> str:
        now = time.monotonic()
        cached = self._cache.get(key_id)
        if cached and (now - cached[0]) < self.ttl_seconds:
            return cached[1]

        actor_url = key_id.split("#", 1)[0]
        try:
            resp = self._client.get(
                actor_url,
                headers={"Accept": "application/activity+json"},
                follow_redirects=True,
            )
            resp.raise_for_status()
        except httpx.HTTPError as exc:
            raise SignatureInvalid(
                f"Impossible de résoudre l'acteur {actor_url!r}: {exc}"
            ) from exc

        data = resp.json()
        pubkey = data.get("publicKey") or {}
        pem = pubkey.get("publicKeyPem")
        if not pem:
            raise SignatureInvalid(
                f"L'acteur {actor_url!r} n'expose pas publicKey.publicKeyPem"
            )
        self._cache[key_id] = (now, pem)
        return pem


@dataclass
class _SignatureMeta:
    """Métadonnées extraites du header Signature."""

    key_id: str
    algorithm: str
    headers: list[str]   # liste des headers couverts par la signature
    signature_b64: str


def _parse_meta(raw: str) -> _SignatureMeta | None:
    import re

    parts = dict(re.findall(r'(\w+)="([^"]*)"', raw or ""))
    if "keyId" not in parts or "signature" not in parts:
        return None
    headers_str = parts.get("headers", "(request-target) host date")
    return _SignatureMeta(
        key_id=parts["keyId"],
        algorithm=parts.get("algorithm", "rsa-sha256"),
        headers=headers_str.split(),
        signature_b64=parts["signature"],
    )


def _build_signing_string(request: Request, headers: list[str]) -> str:
    """Reconstruit la chaîne canonique signée.

    Format : chaque ligne `"<header>: <value>"`, séparées par `\\n`. Pour
    `(request-target)` : `"(request-target): <method-lower> <path-with-query>"`.
    """
    lines: list[str] = []
    for h in headers:
        h_low = h.lower()
        if h_low == "(request-target)":
            target = request.url.path
            if request.url.query:
                target = f"{target}?{request.url.query}"
            lines.append(f"(request-target): {request.method.lower()} {target}")
        else:
            value = request.headers.get(h_low)
            if value is None:
                raise SignatureInvalid(
                    f"Header {h!r} listé dans la signature mais absent de la requête"
                )
            lines.append(f"{h_low}: {value}")
    return "\n".join(lines)


def _verify_date_freshness(request: Request, max_age_seconds: int) -> None:
    date_header = request.headers.get("date")
    if not date_header:
        raise SignatureInvalid("Header Date requis en mode strict")
    try:
        request_time = parsedate_to_datetime(date_header)
    except (TypeError, ValueError) as exc:
        raise SignatureInvalid(f"Header Date non parsable: {date_header!r}") from exc

    if request_time.tzinfo is None:
        request_time = request_time.replace(tzinfo=timezone.utc)
    age = (datetime.now(tz=timezone.utc) - request_time).total_seconds()
    if age > max_age_seconds:
        raise SignatureInvalid(
            f"Requête trop ancienne ({int(age)}s > {max_age_seconds}s, anti-replay)"
        )
    if age < -max_age_seconds:
        raise SignatureInvalid(
            f"Requête datée dans le futur ({int(age)}s, horloge désynchronisée)"
        )


def _verify_signature(public_key_pem: str, signing_string: str, signature_b64: str) -> None:
    try:
        public_key = serialization.load_pem_public_key(public_key_pem.encode("utf-8"))
    except Exception as exc:
        raise SignatureInvalid(f"Clé publique non chargeable: {exc}") from exc

    if not isinstance(public_key, rsa.RSAPublicKey):
        raise SignatureInvalid("La clé publique n'est pas RSA — algorithme non supporté")

    try:
        signature = base64.b64decode(signature_b64)
    except Exception as exc:
        raise SignatureInvalid("Signature non décodable en base64") from exc

    try:
        public_key.verify(
            signature,
            signing_string.encode("utf-8"),
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
    except InvalidSignature as exc:
        raise SignatureInvalid("Signature RSA-SHA256 invalide") from exc


def verify_signature_strict(
    request: Request,
    *,
    key_resolver: KeyResolver,
    max_age_seconds: int = 300,
) -> ParsedSignature:
    """Vérifie cryptographiquement la signature d'une requête entrante.

    Renvoie le `ParsedSignature` (compatible avec le stub) en cas de succès.
    Lève `SignatureInvalid` (HTTPException 401) sinon.
    """
    raw = request.headers.get("signature")
    if not raw:
        raise SignatureInvalid("Header Signature absent")

    meta = _parse_meta(raw)
    if meta is None:
        raise SignatureInvalid("Header Signature non parsable")

    if meta.algorithm.lower() not in ("rsa-sha256", "hs2019"):
        raise SignatureInvalid(
            f"Algorithme {meta.algorithm!r} non supporté (attendu rsa-sha256)"
        )

    _verify_date_freshness(request, max_age_seconds)

    signing_string = _build_signing_string(request, meta.headers)
    public_key_pem = key_resolver.resolve(meta.key_id)
    _verify_signature(public_key_pem, signing_string, meta.signature_b64)

    logger.info("signature OK pour keyId=%s headers=%s", meta.key_id, meta.headers)
    return ParsedSignature(
        key_id=meta.key_id,
        algorithm=meta.algorithm,
        signature_b64=meta.signature_b64,
        raw=raw,
    )


def make_strict_dependency(
    key_resolver: KeyResolver,
    *,
    max_age_seconds: int = 300,
) -> Callable[[Request], ParsedSignature]:
    """Construit une dependency FastAPI qui vérifie strictement la signature."""

    def _dep(request: Request) -> ParsedSignature:
        return verify_signature_strict(
            request,
            key_resolver=key_resolver,
            max_age_seconds=max_age_seconds,
        )

    return _dep
