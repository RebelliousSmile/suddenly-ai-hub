"""T21 — Serveur HTTP FastAPI du service Muses.

Endpoints exposés pour le MVP M2 :
- POST /v1/suggest/dialogue — feature `suggest_dialogue` (#77 côté Suddenly)
- GET /v1/health — liveness check sans auth

L'orchestrateur et l'encodeur sont injectés à la création de l'app pour
permettre les tests (StubEncoder + tables temporaires) et le déploiement
réel (SentenceTransformerEncoder + tables persistées).
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException

from muses.api.auth import ParsedSignature, require_signature_stub
from muses.api.schemas import SuggestRequest, SuggestResponse, SuggestionItem
from muses.pipeline.orchestrator import Orchestrator
from muses.tables.embeddings import Encoder, StubEncoder


def create_app(
    *,
    tables: list[Path],
    encoder: Encoder | None = None,
) -> FastAPI:
    """Construit l'application FastAPI avec un orchestrateur injecté.

    `tables` : liste des chemins JSONL des tables à servir.
    `encoder` : encodeur pour les embeddings du contexte. None = StubEncoder
    (pour les tests). Production utilise SentenceTransformerEncoder.
    """
    encoder = encoder or StubEncoder(dim=16)
    orchestrator = Orchestrator(tables=tables, encoder=encoder)

    app = FastAPI(
        title="Muses",
        description="Service mutualisé d'assistance créative pour le Fediverse Suddenly",
        version="0.0.0-pre-mvp",
    )

    @app.get("/v1/health")
    def health() -> dict:
        return {
            "status": "ok",
            "tables_count": len(tables),
            "encoder_dim": encoder.dim,
        }

    @app.post("/v1/suggest/dialogue", response_model=SuggestResponse)
    def suggest_dialogue(
        req: SuggestRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> SuggestResponse:
        if req.feature != "dialogue":
            raise HTTPException(
                status_code=400,
                detail=f"This endpoint serves only feature='dialogue', got {req.feature!r}",
            )
        result = orchestrator.generate(
            context_text=req.context_text,
            context_tags=req.context_tags,
            n_candidates=req.n_candidates,
            top_n=req.top_n,
        )
        return SuggestResponse(
            suggestions=[
                SuggestionItem(
                    text=c.text,
                    source_row_ids=c.source_row_ids,
                    source_scores=c.source_scores,
                )
                for c in result.candidates
            ],
            relaxed_axes=result.relaxed_axes,
            selected_table_count=len(result.selected_tables),
            weighted_count=result.weighted_count,
        )

    return app
