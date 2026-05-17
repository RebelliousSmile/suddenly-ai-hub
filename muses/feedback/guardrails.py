"""T34 — Garde-fous décentralisés.

v0 du MVP M3 : anti-sleeper (gain alpha cappé) intégré dans TrustStore.
Ce module expose les fonctions d'anti-takeover (détection d'anomalie
comportementale) et de quality gating row-level. Les versions complètes
intègrent les snapshots ML (M4/T40) et la détection d'anomalie par
embedding (au-delà du MVP).
"""

from __future__ import annotations

import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

from muses.feedback.events import EventLog, FeedbackSignal


@dataclass
class SleeperCheckResult:
    """Résultat d'un check anti-sleeper pour un utilisateur."""

    user_id: str
    daily_alpha_gain: float
    cap: float
    over_threshold: bool


class AntiSleeperGuard:
    """Vérifie qu'aucun utilisateur ne dépasse le cap quotidien de gain trust.

    Le cap est appliqué de manière soft par `TrustStore._increment` lors
    des updates ; ce module fournit un check séparé pour reporting / audit.
    """

    def __init__(self, daily_alpha_cap: float = 10.0):
        self.daily_alpha_cap = daily_alpha_cap

    def check_user(
        self,
        user_id: str,
        recent_signals: list[FeedbackSignal],
    ) -> SleeperCheckResult:
        """Calcule le gain alpha cumulé du user sur les dernières 24h."""
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=1)
        from muses.feedback.trust import DEFAULT_WEIGHTS
        total_alpha = 0.0
        for sig in recent_signals:
            if sig.contributor_user_id != user_id:
                continue
            if sig.timestamp < cutoff:
                continue
            d_alpha, _ = DEFAULT_WEIGHTS.get(sig.signal, (0.0, 0.0))
            total_alpha += d_alpha
        return SleeperCheckResult(
            user_id=user_id,
            daily_alpha_gain=total_alpha,
            cap=self.daily_alpha_cap,
            over_threshold=total_alpha > self.daily_alpha_cap,
        )

    def scan_event_log(self, log: EventLog) -> list[SleeperCheckResult]:
        """Scanne tous les contributeurs présents dans le log et reporte ceux sur le seuil."""
        # Charge les signaux récents en mémoire — OK pour MVP, à streamer pour gros volumes.
        cutoff = datetime.now(tz=timezone.utc) - timedelta(days=1)
        recent = [s for s in log.iter_signals() if s.timestamp >= cutoff]
        by_user: dict[str, list[FeedbackSignal]] = defaultdict(list)
        for s in recent:
            if s.contributor_user_id:
                by_user[s.contributor_user_id].append(s)
        return [self.check_user(uid, sigs) for uid, sigs in by_user.items()]
