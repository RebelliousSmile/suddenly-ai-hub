"""T38 — Mode dégradé côté MusesClient."""

import httpx
import pytest

from muses.client import MusesClient, MusesUnavailable
from muses.schemas.tags import AxialTags


def _mock_transport(*, status_code: int = 200, raise_exc: Exception | None = None):
    def handler(request: httpx.Request) -> httpx.Response:
        if raise_exc:
            raise raise_exc
        return httpx.Response(status_code, json={"suggestions": []})
    return httpx.MockTransport(handler)


def _client(transport: httpx.MockTransport) -> MusesClient:
    return MusesClient(http_client=httpx.Client(base_url="http://test", transport=transport))


def test_5xx_raises_muses_unavailable():
    client = _client(_mock_transport(status_code=503))
    with pytest.raises(MusesUnavailable, match="503"):
        client.suggest(
            feature="dialogue",
            context_text="x",
            context_tags=AxialTags(),
            signature='keyId="x",signature="y"',
        )
    client.close()


def test_timeout_raises_muses_unavailable():
    client = _client(_mock_transport(raise_exc=httpx.ReadTimeout("timeout")))
    with pytest.raises(MusesUnavailable, match="injoignable"):
        client.suggest(
            feature="dialogue",
            context_text="x",
            context_tags=AxialTags(),
            signature='keyId="x",signature="y"',
        )
    client.close()


def test_connect_error_raises_muses_unavailable():
    client = _client(_mock_transport(raise_exc=httpx.ConnectError("refused")))
    with pytest.raises(MusesUnavailable, match="injoignable"):
        client.suggest(
            feature="dialogue",
            context_text="x",
            context_tags=AxialTags(),
            signature='keyId="x",signature="y"',
        )
    client.close()


def test_4xx_propagated_not_dégradé():
    client = _client(_mock_transport(status_code=401))
    with pytest.raises(httpx.HTTPStatusError):
        client.suggest(
            feature="dialogue",
            context_text="x",
            context_tags=AxialTags(),
            signature='keyId="x",signature="y"',
        )
    client.close()
