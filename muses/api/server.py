"""T21 — Serveur HTTP FastAPI du service Muses.

Endpoints exposés :
- GET /v1/health — liveness check sans auth
- POST /v1/suggest/dialogue — feature `suggest_dialogue` (#77) [signature requise]
- POST /v1/feedback/signal — capture des 5 signaux UI (M3 T26) [signature requise]
- GET /v1/admin/coverage — carte de couverture admin (M3 T35) [token admin]

L'orchestrateur, l'encodeur, les stores et l'event log sont injectés à la
création de l'app pour permettre les tests et le déploiement.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel, Field

from muses.api.admin import create_admin_router
from muses.api.auth import ParsedSignature, require_signature_stub
from muses.api.schemas import SuggestRequest, SuggestResponse, SuggestionItem
from muses.feedback.events import SIGNAL_TYPES, EventLog, FeedbackSignal, SignalType
from muses.feedback.instance_reputation import InstanceReputationStore
from muses.feedback.online_learning import OnlineLearner
from muses.feedback.style_profile import StyleProfileStore
from muses.feedback.trust import TrustStore
from muses.pipeline.orchestrator import Orchestrator
from muses.tables.embeddings import Encoder, StubEncoder


class FeedbackSignalRequest(BaseModel):
    """Payload du POST /v1/feedback/signal."""

    signal: SignalType
    user_id: str
    instance_id: str
    feature: str
    row_id: str
    contributor_user_id: str | None = None
    contributor_instance_id: str | None = None
    context_tags: dict[str, list[str]] = Field(default_factory=dict)
    edited_text: str | None = None


def create_app(
    *,
    tables: list[Path],
    encoder: Encoder | None = None,
    event_log_path: Path | None = None,
    trust_db_path: Path | None = None,
    instance_db_path: Path | None = None,
    style_db_path: Path | None = None,
    learner_db_path: Path | None = None,
    admin_token: str | None = None,
) -> FastAPI:
    """Construit l'application FastAPI avec orchestrateur et feedback stores.

    Les chemins de persistance sont optionnels : si None, on utilise des
    paths sentinelles `:memory:` (équivalent en pratique : fichiers tmp
    par fixture en tests).
    """
    encoder = encoder or StubEncoder(dim=16)
    orchestrator = Orchestrator(tables=tables, encoder=encoder)

    event_log = EventLog(event_log_path) if event_log_path else None
    trust_store = TrustStore(trust_db_path) if trust_db_path else None
    instance_store = InstanceReputationStore(instance_db_path) if instance_db_path else None
    style_store = StyleProfileStore(style_db_path) if style_db_path else None
    learner = OnlineLearner(learner_db_path) if learner_db_path else None

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
            "feedback_enabled": event_log is not None,
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

    @app.post("/v1/feedback/signal")
    def feedback_signal(
        req: FeedbackSignalRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> dict:
        if event_log is None:
            raise HTTPException(
                status_code=503,
                detail="Feedback log not configured on this instance",
            )
        signal = FeedbackSignal(
            signal=req.signal,
            user_id=req.user_id,
            instance_id=req.instance_id,
            feature=req.feature,
            row_id=req.row_id,
            contributor_user_id=req.contributor_user_id,
            contributor_instance_id=req.contributor_instance_id,
            context_tags=req.context_tags,
            edited_text=req.edited_text,
        )
        event_log.append(signal)

        # Pipeline online : trust + learner + profil
        if trust_store is not None:
            trust_store.update_from_signal(signal)
        if learner is not None:
            learner.update_from_signal(signal)
        if style_store is not None and signal.signal in ("accept", "accept_edited"):
            style_store.observe(
                user_id=signal.user_id,
                row_id=signal.row_id,
                text=signal.edited_text,
            )

        return {"recorded": True, "signal": signal.signal}

    if tables:
        app.include_router(create_admin_router(tables=tables, admin_token=admin_token))

    return app
