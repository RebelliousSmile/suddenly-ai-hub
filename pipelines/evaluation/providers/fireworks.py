"""Fireworks.ai provider implementation.

Uses httpx sync client for API calls. Implements the BaseProvider interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import httpx

from .base import BaseProvider, ChatMessage, CompletionRequest, CompletionResponse


@dataclass
class FireworksProvider(BaseProvider):
    """Provider for Fireworks.ai inference API."""

    api_key: str
    base_url: ClassVar[str] = "https://api.fireworks.ai/inference/v1"
    default_model: ClassVar[str] = "accounts/fireworks/models/qwen2.5-7b-instruct"
    timeout: float = 60.0

    @property
    def name(self) -> str:
        return "fireworks"

    def validate(self) -> bool:
        # Fireworks keys are opaque strings; check non-empty and reasonable length
        return bool(self.api_key and len(self.api_key) > 5)

    def chat_completion(self, request: CompletionRequest) -> CompletionResponse:
        with httpx.Client(
            base_url=self.base_url,
            timeout=httpx.Timeout(self.timeout),
        ) as client:
            payload = {
                "model": request.model,
                "messages": [
                    {"role": m.role, "content": m.content} for m in request.messages
                ],
                "temperature": request.temperature,
                "max_tokens": request.max_tokens,
            }
            resp = client.post(
                "/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
            )
            resp.raise_for_status()
            data = resp.json()
            choice = data["choices"][0]
            return CompletionResponse(
                content=choice["message"]["content"],
                model=data.get("model", request.model),
                usage=data.get("usage"),
            )
