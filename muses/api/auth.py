"""T22 — Stub d'authentification par signature HTTP ActivityPub.

Pour le MVP M2 : on vérifie juste qu'un header `Signature` est présent et
parsable comme une signature HTTP RFC 9421 / ActivityPub draft-cavage.
La vérification cryptographique réelle (résolution de l'acteur, fetch de
la clé publique, validation du digest) viendra dans `infrastructure.md`
au M4/T39.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from fastapi import Header, HTTPException, status


_SIG_PATTERN = re.compile(r'(\w+)="([^"]*)"')


@dataclass
class ParsedSignature:
    """Représentation parsée d'un header Signature ActivityPub draft-cavage."""

    key_id: str
    algorithm: str
    signature_b64: str
    raw: str


def parse_http_signature(raw: str) -> ParsedSignature | None:
    """Parse un header Signature au format draft-cavage. None si invalide."""
    if not raw or not raw.strip():
        return None
    parts = dict(_SIG_PATTERN.findall(raw))
    if "keyId" not in parts or "signature" not in parts:
        return None
    return ParsedSignature(
        key_id=parts["keyId"],
        algorithm=parts.get("algorithm", "rsa-sha256"),
        signature_b64=parts["signature"],
        raw=raw,
    )


def require_signature_stub(
    signature: str | None = Header(None, alias="Signature"),
) -> ParsedSignature:
    """Dependency FastAPI : exige un header Signature parsable.

    **N'effectue PAS la vérification cryptographique.** Voir module docstring.
    """
    parsed = parse_http_signature(signature or "")
    if parsed is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing or unparsable HTTP Signature header",
        )
    return parsed
