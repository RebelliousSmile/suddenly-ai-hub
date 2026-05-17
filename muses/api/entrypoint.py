"""Point d'entrée ASGI prêt pour `uvicorn muses.api.entrypoint:app`.

Lit la configuration depuis l'environnement (cf. `.env.example`), initialise
le logging, construit l'encodeur et l'app FastAPI. Expose `app` au niveau
module pour que les serveurs ASGI standards le trouvent.

Toute erreur de configuration au démarrage lève `ConfigError` et empêche
le démarrage — fail-fast préférable à un service qui sert silencieusement
des résultats incorrects.
"""

from __future__ import annotations

import logging

from muses.api.server import create_app
from muses.config import Settings, configure_logging, load_config
from muses.tables.embeddings import (
    DEFAULT_MODEL,
    Encoder,
    SentenceTransformerEncoder,
    StubEncoder,
)


def _build_encoder(settings: Settings) -> Encoder:
    if settings.encoder == "sentence_transformer":
        model = settings.encoder_model or DEFAULT_MODEL
        return SentenceTransformerEncoder(model_name=model)
    return StubEncoder(dim=settings.stub_encoder_dim)


def _build_app(settings: Settings):
    logger = logging.getLogger("muses.entrypoint")
    table_paths = settings.table_jsonl_paths
    if not table_paths:
        logger.warning(
            "Aucun fichier .jsonl trouvé sous %s — le service répondra "
            "0 suggestion sur toutes les requêtes. Bootstrap manquant ?",
            settings.table_dir,
        )
    logger.info(
        "Démarrage Muses : tables=%d, encoder=%s, signature_mode=%s, admin=%s, rate_limit=%d/min",
        len(table_paths),
        settings.encoder,
        settings.signature_mode,
        "set" if settings.admin_token else "OPEN",
        settings.rate_limit_per_minute,
    )

    return create_app(
        tables=table_paths,
        encoder=_build_encoder(settings),
        event_log_path=settings.feedback_dir / "events.jsonl",
        trust_db_path=settings.feedback_dir / "trust.sqlite",
        instance_db_path=settings.feedback_dir / "instance_reputation.sqlite",
        style_db_path=settings.feedback_dir / "style_profile.sqlite",
        learner_db_path=settings.feedback_dir / "online_learner.sqlite",
        admin_token=settings.admin_token,
        signature_mode=settings.signature_mode,
        signature_max_age_seconds=settings.signature_max_age_seconds,
        rate_limit_per_minute=settings.rate_limit_per_minute,
    )


# Chargé à l'import — uvicorn appelle `import muses.api.entrypoint`
# et utilise `app`. Toute erreur de config remonte ici → uvicorn refuse
# de démarrer, comportement souhaité.
_settings = load_config()
configure_logging(_settings)
app = _build_app(_settings)
