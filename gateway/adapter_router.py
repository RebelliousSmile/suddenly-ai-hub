from __future__ import annotations

from typing import Optional

from .config import get_config


def resolve_adapter(genre: Optional[str], situation: Optional[str]) -> Optional[str]:
    """Return adapter name for vLLM, or None to use the base model."""
    adapters = get_config().available_adapters

    candidates = []
    if genre and situation:
        candidates.append(f"lora-{genre}-{situation}")
    if situation:
        candidates.append(f"lora-{situation}")
    if genre:
        candidates.append(f"lora-{genre}")
    candidates.append("lora-generique")

    for name in candidates:
        if name in adapters:
            return name
    return None
