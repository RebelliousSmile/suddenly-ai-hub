from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Final

# ---------------------------------------------------------------------------
# Taxonomie statique — sera chargée dynamiquement depuis la DB en Phase 1
# Genres = axe univers (GROG)
# ---------------------------------------------------------------------------

GENRES_DEFAULT: Final[list[str]] = [
    "medieval-fantastique",
    "historique-fantastique",
    "scifi",
    "contemporain-fantastique",
    "space-opera",
    "contemporain",
    "post-apocalyptique",
    "cyberpunk",
    "super-heros",
    "oriental-manga",
    "generique",
]

SITUATIONS_DEFAULT: Final[list[str]] = [
    "combat",
    "exploration",
    "dialogue",
    "romance",
    "intrigue",
    "repos",
    "voyage",
]

MODELS_DEFAULT: Final[list[str]] = [
    "suddenly-7b",
    "suddenly-13b",
]


@dataclass
class GatewayConfig:
    genres: list[str] = field(default_factory=lambda: list(GENRES_DEFAULT))
    situations: list[str] = field(default_factory=lambda: list(SITUATIONS_DEFAULT))
    models: list[str] = field(default_factory=lambda: list(MODELS_DEFAULT))
    # Adapters réellement chargés dans vLLM (vide = aucun LoRA actif)
    available_adapters: frozenset[str] = field(default_factory=frozenset)
    vllm_base_url: str = field(default_factory=lambda: os.environ.get("VLLM_BASE_URL", "http://localhost:8001"))
    default_model: str = "suddenly-7b"


# Instance globale — remplacée au démarrage via lifespan ou dans les tests
_config: GatewayConfig = GatewayConfig()


def get_config() -> GatewayConfig:
    return _config


def set_config(cfg: GatewayConfig) -> None:
    global _config
    _config = cfg
