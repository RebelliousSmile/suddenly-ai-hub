from __future__ import annotations

from typing import Optional

import httpx

from .config import get_config
from .models import ChatRequest, ChatChoice, ChatResponse, ChatUsage, Message


async def chat(request: ChatRequest, adapter: Optional[str] = None) -> ChatResponse:
    cfg = get_config()

    payload: dict = {
        "model": adapter if adapter else request.model,
        "messages": [m.model_dump() for m in request.messages],
        "max_tokens": request.max_tokens,
        "temperature": request.temperature,
        "stream": False,
    }

    async with httpx.AsyncClient(base_url=cfg.vllm_base_url, timeout=60.0) as client:
        resp = await client.post("/v1/chat/completions", json=payload)
        resp.raise_for_status()
        data = resp.json()

    choice = data["choices"][0]
    return ChatResponse(
        id=data["id"],
        object="chat.completion",
        model=request.model,
        adapter_used=adapter,
        choices=[
            ChatChoice(
                index=0,
                message=Message(role="assistant", content=choice["message"]["content"]),
                finish_reason=choice.get("finish_reason"),
            )
        ],
        usage=ChatUsage(
            prompt_tokens=data["usage"]["prompt_tokens"],
            completion_tokens=data["usage"]["completion_tokens"],
            total_tokens=data["usage"]["total_tokens"],
        ),
    )
