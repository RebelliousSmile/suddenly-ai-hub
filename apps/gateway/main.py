from __future__ import annotations

import uuid
from contextlib import asynccontextmanager
from typing import Optional

import httpx
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi import Depends, FastAPI, HTTPException
import boto3
from botocore.config import Config as BotoConfig

from . import vllm_client
from .adapter_router import resolve_adapter
from .auth import clear_key_cache, preload_key, verify_http_signature
from .config import get_config
from .models import (
    ChatRequest,
    ChatResponse,
    ContributeRequest,
    ContributeResponse,
    HealthResponse,
    ModelInfo,
    ModelsResponse,
    StatsResponse,
)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = get_config()
    if cfg.activitypub_mock:
        private_key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
        public_key = private_key.public_key()
        mock_key_id = f"{cfg.mock_instance_url}/actor#main-key"
        preload_key(mock_key_id, public_key)
        app.state.mock_private_key = private_key
        app.state.mock_key_id = mock_key_id
    yield
    clear_key_cache()


app = FastAPI(title="Suddenly AI Hub Gateway", lifespan=lifespan)


def _validate_genre(genre: Optional[str]) -> None:
    if genre is None:
        return
    valid = get_config().genres
    if genre not in valid:
        raise HTTPException(
            status_code=422,
            detail={"message": f"Genre inconnu : {genre!r}", "valid_values": valid},
        )


def _validate_situation(situation: Optional[str]) -> None:
    if situation is None:
        return
    valid = get_config().situations
    if situation not in valid:
        raise HTTPException(
            status_code=422,
            detail={"message": f"Situation inconnue : {situation!r}", "valid_values": valid},
        )


@app.post("/v1/chat/completions", response_model=ChatResponse)
async def chat_completions(
    request: ChatRequest,
    _: None = Depends(verify_http_signature),
):
    _validate_genre(request.genre)
    _validate_situation(request.situation)
    adapter = resolve_adapter(request.genre, request.situation)
    try:
        return await vllm_client.chat(request, adapter)
    except httpx.HTTPStatusError as exc:
        raise HTTPException(status_code=502, detail=f"vLLM error: {exc.response.status_code}")
    except httpx.RequestError as exc:
        raise HTTPException(status_code=503, detail=f"vLLM unreachable: {exc}")


@app.get("/v1/models", response_model=ModelsResponse)
def list_models():
    cfg = get_config()
    return ModelsResponse(
        data=[
            ModelInfo(id=m, type="base", available_adapters=sorted(cfg.available_adapters))
            for m in cfg.models
        ]
    )


@app.get("/v1/health", response_model=HealthResponse)
async def health():
    cfg = get_config()
    vllm_ok = False
    try:
        async with httpx.AsyncClient(base_url=cfg.vllm_base_url, timeout=5.0) as client:
            resp = await client.get("/health")
            vllm_ok = resp.is_success
    except Exception:
        pass

    s3_ok = None
    s3_error = None
    if cfg.s3_endpoint:
        s3_ok = False
        try:
            s3 = boto3.client(
                "s3",
                endpoint_url=cfg.s3_endpoint,
                aws_access_key_id=cfg.s3_access_key,
                aws_secret_access_key=cfg.s3_secret_key,
                region_name=cfg.s3_region,
                config=BotoConfig(
                    signature_version="s3v4",
                    s3={"addressing_style": "path"},
                ),
            )
            s3.list_objects_v2(Bucket=cfg.s3_bucket, MaxKeys=1)
            s3_ok = True
        except Exception as exc:
            import logging
            logging.getLogger("gateway.health").warning("S3 healthcheck failed: %s", exc)
            s3_error = str(exc)

    if vllm_ok and (s3_ok is None or s3_ok):
        status = "ok"
    elif not vllm_ok and s3_ok is None:
        # S3 not configured, vLLM down = degraded (not error)
        status = "degraded"
    elif vllm_ok or s3_ok:
        status = "degraded"
    else:
        status = "error"

    return HealthResponse(
        status=status,
        vllm_reachable=vllm_ok,
        models_loaded=len(cfg.models) if vllm_ok else 0,
        s3_reachable=s3_ok,
        s3_error=s3_error,
    )


@app.post("/v1/contribute", response_model=ContributeResponse)
async def contribute(
    request: ContributeRequest,
    _: None = Depends(verify_http_signature),
):
    _validate_genre(request.genre)
    _validate_situation(request.situation)
    return ContributeResponse(
        accepted=True,
        session_id=str(uuid.uuid4()),
        message="Session reçue et mise en file d'attente.",
    )


@app.get("/v1/stats", response_model=StatsResponse)
def stats():
    cfg = get_config()
    return StatsResponse(
        models_available=len(cfg.models),
        adapters_active=len(cfg.available_adapters),
        sessions_contributed=0,
    )
