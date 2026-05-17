"""T28 — Réputation d'instance + multiplicateur global.

Voir learning-and-trust.md §5. Multiplicateur borné dans [0.3, 1.2].
Calcul automatique = base + agrégation des signaux ; override admin
possible.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal


MULTIPLIER_MIN = 0.3
MULTIPLIER_MAX = 1.2
MULTIPLIER_DEFAULT = 1.0


@dataclass
class InstanceReputation:
    instance_id: str
    base_multiplier: float
    reviewed_at: datetime
    source: Literal["auto", "admin"]


_SCHEMA = """
CREATE TABLE IF NOT EXISTS instance_reputation (
    instance_id TEXT PRIMARY KEY,
    base_multiplier REAL NOT NULL,
    reviewed_at TEXT NOT NULL,
    source TEXT NOT NULL
);
"""


def clamp_multiplier(value: float) -> float:
    return max(MULTIPLIER_MIN, min(MULTIPLIER_MAX, value))


class InstanceReputationStore:
    """Persistance SQLite des multiplicateurs par instance."""

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def get(self, instance_id: str) -> InstanceReputation:
        """Renvoie la réputation. Default = neutre (1.0, source=auto)."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT base_multiplier, reviewed_at, source "
                "FROM instance_reputation WHERE instance_id=?",
                (instance_id,),
            ).fetchone()
        if row is None:
            return InstanceReputation(
                instance_id=instance_id,
                base_multiplier=MULTIPLIER_DEFAULT,
                reviewed_at=datetime.now(tz=timezone.utc),
                source="auto",
            )
        return InstanceReputation(
            instance_id=instance_id,
            base_multiplier=row[0],
            reviewed_at=datetime.fromisoformat(row[1]),
            source=row[2],
        )

    def set_admin_override(self, instance_id: str, multiplier: float) -> None:
        """Override admin explicite. Borne automatiquement le multiplicateur."""
        m = clamp_multiplier(multiplier)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO instance_reputation "
                "(instance_id, base_multiplier, reviewed_at, source) "
                "VALUES (?, ?, ?, ?)",
                (instance_id, m, datetime.now(tz=timezone.utc).isoformat(), "admin"),
            )
            conn.commit()

    def update_auto(
        self,
        instance_id: str,
        *,
        accept_rate: float,
        n_signals: int,
        prior_strength: float = 50.0,
    ) -> None:
        """Recalcul auto à partir d'un taux d'accept moyen sur N signaux.

        Formule simple : multiplicateur = 0.6 + 0.6 * smoothed_accept_rate
        où smoothed_accept_rate est lissé par un prior neutre à 0.6.
        Mappe [0, 1] → [0.6, 1.2] avec pénalisation pour faible volume.
        """
        smoothed = (
            (accept_rate * n_signals + 0.6 * prior_strength)
            / (n_signals + prior_strength)
        )
        m = clamp_multiplier(0.6 + 0.6 * smoothed)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR REPLACE INTO instance_reputation "
                "(instance_id, base_multiplier, reviewed_at, source) "
                "VALUES (?, ?, ?, ?)",
                (instance_id, m, datetime.now(tz=timezone.utc).isoformat(), "auto"),
            )
            conn.commit()
