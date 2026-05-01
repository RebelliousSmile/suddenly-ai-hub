from __future__ import annotations

from typing import Literal, Optional
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Primitives
# ---------------------------------------------------------------------------

class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


# ---------------------------------------------------------------------------
# POST /v1/chat/completions
# ---------------------------------------------------------------------------

class ChatRequest(BaseModel):
    model: str = "suddenly-7b"
    messages: list[Message] = Field(..., min_length=1)
    genre: Optional[str] = None
    situation: Optional[str] = None
    max_tokens: int = Field(default=512, ge=1, le=4096)
    temperature: float = Field(default=0.8, ge=0.0, le=2.0)
    stream: bool = False


class ChatChoice(BaseModel):
    index: int
    message: Message
    finish_reason: Optional[str] = None


class ChatUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class ChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    model: str
    adapter_used: Optional[str] = None
    choices: list[ChatChoice]
    usage: ChatUsage


# ---------------------------------------------------------------------------
# GET /v1/models
# ---------------------------------------------------------------------------

class ModelInfo(BaseModel):
    id: str
    type: Literal["base", "adapter"]
    available_adapters: list[str] = Field(default_factory=list)


class ModelsResponse(BaseModel):
    object: str = "list"
    data: list[ModelInfo]


# ---------------------------------------------------------------------------
# GET /v1/health
# ---------------------------------------------------------------------------

class HealthResponse(BaseModel):
    status: Literal["ok", "degraded", "error"]
    vllm_reachable: bool
    models_loaded: int


# ---------------------------------------------------------------------------
# POST /v1/contribute
# ---------------------------------------------------------------------------

class ContributeRequest(BaseModel):
    messages: list[Message] = Field(..., min_length=2)
    genre: Optional[str] = None
    situation: Optional[str] = None
    source_instance: str = Field(..., description="URL de l'instance Suddenly source")


class ContributeResponse(BaseModel):
    accepted: bool
    session_id: str
    message: str


# ---------------------------------------------------------------------------
# GET /v1/stats
# ---------------------------------------------------------------------------

class StatsResponse(BaseModel):
    models_available: int
    adapters_active: int
    sessions_contributed: int
