"""T27 — Trust contextuel par auteur : Beta reputation par (user, axe, valeur).

Voir learning-and-trust.md §4. Backed par SQLite pour la persistance simple.
La table `trust` stocke α, β, last_update par triplet (user, axis, value).

Pondération downstream :
- trust_mean = α / (α + β)
- confiance (proxy) = (α + β) / (α + β + prior_strength)
- trust_penalized = 0.5 + (trust_mean - 0.5) * confidence_factor
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from math import exp, log
from pathlib import Path

from muses.feedback.events import FeedbackSignal, SignalType
from muses.schemas.tags import AXIS_NAMES


# Poids par signal (cf. learning-and-trust.md §4 Update)
DEFAULT_WEIGHTS: dict[SignalType, tuple[float, float]] = {
    "accept": (1.0, 0.0),
    "accept_edited": (0.5, 0.0),
    "reject_off": (0.0, 1.0),
    "reject_challenge_appreciated": (0.0, 0.0),  # neutre
    "ignore": (0.0, 0.2),
}


# Anti-sleeper : gain max d'alpha par fenêtre temporelle pour un même
# triplet (user, axis, value). Cf. learning-and-trust.md §6 Anti-sleeper.
DEFAULT_DAILY_ALPHA_CAP = 10.0


# Demi-vie de la décroissance temporelle (cf. learning-and-trust.md §4).
DEFAULT_HALF_LIFE_DAYS = 180.0


@dataclass
class BetaReputation:
    alpha: float
    beta: float
    last_update: datetime

    def decayed(self, now: datetime, half_life_days: float) -> "BetaReputation":
        """Renvoie une copie avec α et β décrus selon le temps écoulé."""
        delta_days = max((now - self.last_update).total_seconds() / 86400.0, 0)
        factor = exp(-log(2) * delta_days / half_life_days)
        return BetaReputation(
            alpha=self.alpha * factor,
            beta=self.beta * factor,
            last_update=self.last_update,
        )

    def mean(self) -> float:
        total = self.alpha + self.beta
        return self.alpha / total if total > 0 else 0.5

    def confidence(self, prior_strength: float = 10.0) -> float:
        """Proxy : (n / (n + prior_strength)) — entre 0 et 1."""
        n = self.alpha + self.beta
        return n / (n + prior_strength)

    def penalized_score(self, prior_strength: float = 10.0) -> float:
        """Trust mean pénalisé par la confiance.

        À faible confiance, le score tend vers 0.5 (prior neutre). À haute
        confiance, vers la moyenne réelle.
        """
        c = self.confidence(prior_strength)
        return 0.5 + (self.mean() - 0.5) * c


_SCHEMA = """
CREATE TABLE IF NOT EXISTS trust (
    user_id TEXT NOT NULL,
    axis TEXT NOT NULL,
    value TEXT NOT NULL,
    alpha REAL NOT NULL DEFAULT 1.0,
    beta REAL NOT NULL DEFAULT 1.0,
    last_update TEXT NOT NULL,
    PRIMARY KEY (user_id, axis, value)
);
"""


class TrustStore:
    """Persistance SQLite des trust scores par (user, axis, value).

    Le store est conçu pour être interrogé à la fois pour les updates
    (à chaque signal) et pour les queries (à chaque pondération downstream).
    """

    def __init__(
        self,
        db_path: Path,
        *,
        half_life_days: float = DEFAULT_HALF_LIFE_DAYS,
        daily_alpha_cap: float = DEFAULT_DAILY_ALPHA_CAP,
    ):
        self.db_path = Path(db_path)
        self.half_life_days = half_life_days
        self.daily_alpha_cap = daily_alpha_cap
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def _now(self) -> datetime:
        return datetime.now(tz=timezone.utc)

    def update_from_signal(self, signal: FeedbackSignal) -> int:
        """Met à jour le trust du contributeur (auteur de la row).

        Renvoie le nombre de triplets (axis, value) touchés.
        """
        if signal.contributor_user_id is None:
            return 0  # bootstrap/mined rows : pas de contributeur à créditer
        weights = DEFAULT_WEIGHTS.get(signal.signal)
        if weights is None or weights == (0.0, 0.0):
            return 0
        d_alpha, d_beta = weights

        ctx_tags = signal.context_tags or {}
        touched = 0
        now = self._now()
        for axis in AXIS_NAMES:
            for value in ctx_tags.get(axis, []):
                self._increment(
                    signal.contributor_user_id, axis, value, d_alpha, d_beta, now,
                )
                touched += 1
        return touched

    def _increment(
        self,
        user_id: str,
        axis: str,
        value: str,
        d_alpha: float,
        d_beta: float,
        now: datetime,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT alpha, beta, last_update FROM trust "
                "WHERE user_id=? AND axis=? AND value=?",
                (user_id, axis, value),
            ).fetchone()

            if row is None:
                # Premier signal sur ce triplet : démarrer du prior Beta(1, 1)
                # avant d'appliquer le delta. C'est ce qui distingue 95%
                # sur 1000 (haute confiance) de 95% sur 5 (faible) — sans
                # prior, on perdrait cette information dès le 1er signal.
                alpha = 1.0 + d_alpha
                beta = 1.0 + d_beta
            else:
                alpha_db, beta_db, last_iso = row
                last_update = datetime.fromisoformat(last_iso)
                # Anti-sleeper : cap le gain quotidien
                if (now - last_update) < timedelta(days=1):
                    d_alpha_effective = min(d_alpha, self.daily_alpha_cap)
                else:
                    d_alpha_effective = d_alpha
                alpha = alpha_db + d_alpha_effective
                beta = beta_db + d_beta

            conn.execute(
                "INSERT OR REPLACE INTO trust "
                "(user_id, axis, value, alpha, beta, last_update) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (user_id, axis, value, alpha, beta, now.isoformat()),
            )
            conn.commit()

    def get(self, user_id: str, axis: str, value: str) -> BetaReputation:
        """Lit le trust pour un triplet. Renvoie prior Beta(1,1) si absent."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT alpha, beta, last_update FROM trust "
                "WHERE user_id=? AND axis=? AND value=?",
                (user_id, axis, value),
            ).fetchone()
        if row is None:
            return BetaReputation(alpha=1.0, beta=1.0, last_update=self._now())
        return BetaReputation(
            alpha=row[0],
            beta=row[1],
            last_update=datetime.fromisoformat(row[2]),
        )

    def penalized_score(self, user_id: str, axis: str, value: str) -> float:
        """Score trust appliqué (avec décroissance + pénalisation confiance)."""
        rep = self.get(user_id, axis, value)
        decayed = rep.decayed(self._now(), self.half_life_days)
        return decayed.penalized_score()
