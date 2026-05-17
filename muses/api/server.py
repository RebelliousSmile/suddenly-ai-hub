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

from muses.analysis.coherence import (
    analyze_scene_coherence,
    analyze_session_coherence,
)
from muses.analysis.federated_links import find_federated_links
from muses.analysis.summary import generate_session_summary
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


class SceneCoherenceRequest(BaseModel):
    scene_fragments: list[str]


class SessionCoherenceRequest(BaseModel):
    scenes: list[list[str]]


class FederatedLinksRequest(BaseModel):
    session_characters: dict[str, str]
    public_characters: dict[str, str]
    threshold: float = Field(0.3, ge=0.0, le=1.0)


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

    GENERATION_FEATURES = {"dialogue", "action", "description", "thought", "video_prompt"}

    def _serve_generation(req: SuggestRequest, expected: str) -> SuggestResponse:
        if req.feature != expected:
            raise HTTPException(
                status_code=400,
                detail=f"This endpoint serves only feature={expected!r}, got {req.feature!r}",
            )
        result = orchestrator.generate(
            context_text=req.context_text,
            context_tags=req.context_tags,
            n_candidates=req.n_candidates,
            top_n=req.top_n,
            mode=req.mode,
            user_id=req.user_id,
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

    @app.post("/v1/suggest/dialogue", response_model=SuggestResponse)
    def suggest_dialogue(
        req: SuggestRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> SuggestResponse:
        return _serve_generation(req, "dialogue")

    @app.post("/v1/suggest/action", response_model=SuggestResponse)
    def suggest_action(
        req: SuggestRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> SuggestResponse:
        return _serve_generation(req, "action")

    @app.post("/v1/suggest/description", response_model=SuggestResponse)
    def suggest_description(
        req: SuggestRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> SuggestResponse:
        return _serve_generation(req, "description")

    @app.post("/v1/suggest/thought", response_model=SuggestResponse)
    def suggest_thought(
        req: SuggestRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> SuggestResponse:
        return _serve_generation(req, "thought")

    @app.post("/v1/suggest/video_prompt", response_model=SuggestResponse)
    def suggest_video_prompt(
        req: SuggestRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> SuggestResponse:
        # T44 — pas de filtre best-of-N pour cette feature (cf. use-cases.md §4.4)
        # Pour MVP M5, on garde le même pipeline ; la spécialisation
        # canvas-fixe vient avec la curation dédiée des templates visuels.
        return _serve_generation(req, "video_prompt")

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

    # --- Endpoints d'analyse (M5/T47-T50) -----------------------------------

    @app.post("/v1/analyze/consistency_scene")
    def analyze_consistency_scene_endpoint(
        req: SceneCoherenceRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> dict:
        issues = analyze_scene_coherence(req.scene_fragments)
        return {
            "n_issues": len(issues),
            "issues": [
                {"severity": i.severity, "fragment_index": i.fragment_index,
                 "description": i.description}
                for i in issues
            ],
        }

    @app.post("/v1/analyze/consistency_session")
    def analyze_consistency_session_endpoint(
        req: SessionCoherenceRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> dict:
        return analyze_session_coherence(req.scenes)

    @app.post("/v1/analyze/summary")
    def analyze_summary_endpoint(
        req: SessionCoherenceRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> dict:
        return {"summary": generate_session_summary(req.scenes)}

    @app.post("/v1/analyze/federated_links")
    def analyze_federated_links_endpoint(
        req: FederatedLinksRequest,
        sig: ParsedSignature = Depends(require_signature_stub),
    ) -> dict:
        suggestions = find_federated_links(
            req.session_characters,
            req.public_characters,
            encoder=encoder,
            threshold=req.threshold,
        )
        return {
            "suggestions": [
                {
                    "session_character": s.session_character,
                    "public_character_id": s.public_character_id,
                    "similarity": s.similarity,
                    "confidence": s.confidence,
                }
                for s in suggestions
            ],
        }

    if tables:
        app.include_router(create_admin_router(tables=tables, admin_token=admin_token))

    return app
