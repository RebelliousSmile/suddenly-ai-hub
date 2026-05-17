"""T32 — Profil de style auteur : histogrammes à 4 niveaux + décroissance.

Voir style-coaching.md §1 Quatre niveaux d'observation. v0 : compteurs en
SQLite avec décroissance temporelle calculée à la lecture (pas in-place).

Niveaux observés :
- Fréquence par row (par row_id que l'auteur a accepté ou réutilisé)
- Fréquence par template (template id)
- Fréquence par beat (beat label)
- Profil lexical (n-grammes — n=1 pour v0)
"""

from __future__ import annotations

import re
import sqlite3
from collections import Counter
from datetime import datetime, timezone
from math import exp, log
from pathlib import Path


DEFAULT_HALF_LIFE_DAYS = 90.0  # demi-vie ~3 mois, cf. style-coaching.md §1


_SCHEMA = """
CREATE TABLE IF NOT EXISTS style_observations (
    user_id TEXT NOT NULL,
    dimension TEXT NOT NULL,   -- 'row' | 'template' | 'beat' | 'lex'
    key TEXT NOT NULL,         -- row_id, template_id, beat_label, ngram
    count REAL NOT NULL DEFAULT 0.0,
    last_update TEXT NOT NULL,
    PRIMARY KEY (user_id, dimension, key)
);
CREATE INDEX IF NOT EXISTS idx_style_user_dim ON style_observations(user_id, dimension);
"""


_TOKEN_PATTERN = re.compile(r"\b[\w'-]+\b", flags=re.UNICODE)


def _decayed(count: float, last_update: datetime, now: datetime, half_life_days: float) -> float:
    delta_days = max((now - last_update).total_seconds() / 86400.0, 0)
    factor = exp(-log(2) * delta_days / half_life_days)
    return count * factor


class StyleProfileStore:
    """Persistance SQLite du profil de style par auteur."""

    def __init__(self, db_path: Path, *, half_life_days: float = DEFAULT_HALF_LIFE_DAYS):
        self.db_path = Path(db_path)
        self.half_life_days = half_life_days
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def observe(
        self,
        user_id: str,
        *,
        row_id: str | None = None,
        template_id: str | None = None,
        beat_label: str | None = None,
        text: str | None = None,
        weight: float = 1.0,
    ) -> None:
        """Enregistre une observation. Plusieurs dimensions possibles en un appel."""
        now = datetime.now(tz=timezone.utc)
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.cursor()
            if row_id:
                self._increment(cur, user_id, "row", row_id, weight, now)
            if template_id:
                self._increment(cur, user_id, "template", template_id, weight, now)
            if beat_label:
                self._increment(cur, user_id, "beat", beat_label, weight, now)
            if text:
                for token in _TOKEN_PATTERN.findall(text.lower()):
                    self._increment(cur, user_id, "lex", token, weight, now)
            conn.commit()

    @staticmethod
    def _increment(cur: sqlite3.Cursor, user_id: str, dim: str, key: str,
                   weight: float, now: datetime) -> None:
        row = cur.execute(
            "SELECT count FROM style_observations "
            "WHERE user_id=? AND dimension=? AND key=?",
            (user_id, dim, key),
        ).fetchone()
        count = (row[0] if row else 0.0) + weight
        cur.execute(
            "INSERT OR REPLACE INTO style_observations "
            "(user_id, dimension, key, count, last_update) "
            "VALUES (?, ?, ?, ?, ?)",
            (user_id, dim, key, count, now.isoformat()),
        )

    def top(
        self,
        user_id: str,
        dimension: str,
        *,
        limit: int = 10,
    ) -> list[tuple[str, float]]:
        """Top-K observations pour une dimension donnée, avec décroissance."""
        now = datetime.now(tz=timezone.utc)
        with sqlite3.connect(self.db_path) as conn:
            rows = conn.execute(
                "SELECT key, count, last_update FROM style_observations "
                "WHERE user_id=? AND dimension=?",
                (user_id, dimension),
            ).fetchall()
        decayed = [
            (key, _decayed(count, datetime.fromisoformat(ts), now, self.half_life_days))
            for key, count, ts in rows
        ]
        decayed.sort(key=lambda x: x[1], reverse=True)
        return decayed[:limit]

    def has_minimum_observations(self, user_id: str, threshold: int = 50) -> bool:
        """Vérifie si l'auteur a suffisamment d'historique pour le mode challenge."""
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT SUM(count) FROM style_observations WHERE user_id=?",
                (user_id,),
            ).fetchone()
        total = row[0] or 0
        return total >= threshold

    def purge_user(self, user_id: str) -> int:
        """Supprime tout l'historique d'un user (RGPD friendly)."""
        with sqlite3.connect(self.db_path) as conn:
            cur = conn.execute(
                "DELETE FROM style_observations WHERE user_id=?", (user_id,),
            )
            conn.commit()
            return cur.rowcount
