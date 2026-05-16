"""Together.ai provider implementation.

Uses httpx sync client for API calls. Implements the BaseProvider interface.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

import httpx

from .base import BaseProvider, ChatMessage, CompletionRequest, CompletionResponse


@dataclass
class TogetherProvider(BaseProvider):
    """Provider for Together.ai inference API."""

    api_key: str
    base_url: ClassVar[str] = "https://api.together.xyz/v1"
    default_model: ClassVar[str] = "Qwen/Qwen2.5-7B-Instruct"
    timeout: float = 60.0

    @property
    def name(self) -> str:
        return "together"

    def validate(self) -> bool:
        # Keys may be "sk-..." (old format) or "tgp_v1-..." (new format)
        # Just check non-empty and reasonable length
        return bool(self.api_key and len(self.api_key) > 10)

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
