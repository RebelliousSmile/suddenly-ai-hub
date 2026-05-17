"""Tests T22 — stub d'auth signature ActivityPub."""

import pytest
from fastapi import HTTPException

from muses.api.auth import parse_http_signature, require_signature_stub


class TestParseSignature:
    def test_parses_canonical_format(self):
        raw = 'keyId="https://exemple.tld/users/alice#main-key",algorithm="rsa-sha256",signature="abc123=="'
        parsed = parse_http_signature(raw)
        assert parsed is not None
        assert parsed.key_id == "https://exemple.tld/users/alice#main-key"
        assert parsed.algorithm == "rsa-sha256"
        assert parsed.signature_b64 == "abc123=="

    def test_missing_keyid_rejected(self):
        raw = 'algorithm="rsa-sha256",signature="abc"'
        assert parse_http_signature(raw) is None

    def test_missing_signature_rejected(self):
        raw = 'keyId="x",algorithm="rsa-sha256"'
        assert parse_http_signature(raw) is None

    def test_empty_rejected(self):
        assert parse_http_signature("") is None
        assert parse_http_signature("   ") is None
        assert parse_http_signature(None) is None

    def test_default_algorithm(self):
        raw = 'keyId="x",signature="abc"'
        parsed = parse_http_signature(raw)
        assert parsed.algorithm == "rsa-sha256"


class TestRequireSignatureStub:
    def test_missing_header_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            require_signature_stub(signature=None)
        assert exc_info.value.status_code == 401

    def test_unparsable_header_raises_401(self):
        with pytest.raises(HTTPException) as exc_info:
            require_signature_stub(signature="garbage")
        assert exc_info.value.status_code == 401

    def test_valid_header_passes(self):
        raw = 'keyId="x",signature="abc"'
        parsed = require_signature_stub(signature=raw)
        assert parsed.key_id == "x"
