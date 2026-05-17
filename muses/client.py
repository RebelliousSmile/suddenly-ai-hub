"""T24 — MusesClient : abstraction côté instance Suddenly.

Voir external/use-cases.md §4.1. Pour le MVP M2, ne supporte que la méthode
`suggest` sur la feature `dialogue`. `analyze` viendra avec le pipeline
d'analyse en M5.
"""

from __future__ import annotations

from dataclasses import dataclass

import httpx

from muses.schemas.tags import AxialTags


@dataclass
class MusesSuggestion:
    """Une suggestion individuelle parsée depuis la réponse Muses."""

    text: str
    source_row_ids: list[str]
    source_scores: list[float]


@dataclass
class MusesSuggestResult:
    """Résultat d'un appel suggest. Inclut la traçabilité globale."""

    suggestions: list[MusesSuggestion]
    relaxed_axes: list[str]
    selected_table_count: int
    weighted_count: int


class MusesClient:
    """Client HTTP minimal côté instance Suddenly.

    L'authentification est faite par signature HTTP ActivityPub. Pour le
    MVP M2, on accepte une signature pré-calculée fournie par le caller
    (le calcul réel — canonicalisation, hash, RSA — est traité dans la
    couche ActivityPub de l'instance, hors périmètre Muses).
    """

    def __init__(
        self,
        base_url: str | None = None,
        *,
        timeout: float = 5.0,
        http_client: httpx.Client | None = None,
    ):
        """Soit `base_url` (le client construit son propre httpx.Client),
        soit `http_client` (injecté — utile pour les tests avec FastAPI's
        TestClient, qui est une sous-classe httpx.Client compatible ASGI).
        """
        if http_client is not None:
            self._client = http_client
            self.base_url = base_url or ""
        else:
            if base_url is None:
                raise ValueError("base_url required when http_client is not provided")
            self.base_url = base_url.rstrip("/")
            self._client = httpx.Client(base_url=self.base_url, timeout=timeout)

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MusesClient":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def health(self) -> dict:
        """Liveness probe — pas d'auth requise."""
        resp = self._client.get("/v1/health")
        resp.raise_for_status()
        return resp.json()

    def suggest(
        self,
        *,
        feature: str,
        context_text: str,
        context_tags: AxialTags,
        signature: str,
        n_candidates: int = 5,
        top_n: int = 3,
    ) -> MusesSuggestResult:
        """Envoie une requête de suggestion. Renvoie le résultat parsé.

        `signature` est le header HTTP Signature draft-cavage complet, déjà
        construit côté instance.
        """
        if feature != "dialogue":
            raise ValueError(f"Feature {feature!r} non supportée par cette version du client")

        payload = {
            "feature": feature,
            "context_text": context_text,
            "context_tags": context_tags.model_dump(),
            "n_candidates": n_candidates,
            "top_n": top_n,
        }
        resp = self._client.post(
            "/v1/suggest/dialogue",
            json=payload,
            headers={"Signature": signature},
        )
        resp.raise_for_status()
        data = resp.json()
        return MusesSuggestResult(
            suggestions=[
                MusesSuggestion(
                    text=s["text"],
                    source_row_ids=s.get("source_row_ids", []),
                    source_scores=s.get("source_scores", []),
                )
                for s in data["suggestions"]
            ],
            relaxed_axes=data.get("relaxed_axes", []),
            selected_table_count=data.get("selected_table_count", 0),
            weighted_count=data.get("weighted_count", 0),
        )
