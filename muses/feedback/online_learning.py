"""T29-T31 — Online learning des étages 1, 2, 4.

**v0 du MVP M3** : interfaces stables avec updates incrémentaux simples.
Les vrais classifieurs ML appris (DPO-lite étage 4, cross-encoder étage 2,
classifieur multi-label étage 1) sont implémentés en M5/T51 — ils
brancheront sur cette même API en consommant l'event log.

Pour M3, on capture les signaux dans des "scores de pertinence" persistés
par (context_axes, row_id) ; le pondérateur peut les utiliser comme
modulateurs.
"""

from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from muses.feedback.events import FeedbackSignal, SignalType


# Poids des signaux pour la pertinence row-en-contexte (étage 4 v0)
# Cf. style-coaching.md §3 colonne "Update ranker étage 4"
SIGNAL_SCORE_DELTAS: dict[SignalType, float] = {
    "accept": 1.0,
    "accept_edited": 0.5,
    "reject_off": -1.0,
    "reject_challenge_appreciated": 0.0,  # neutre ranker, cf. style-coaching §3
    "ignore": -0.2,
}


_SCHEMA = """
CREATE TABLE IF NOT EXISTS context_row_scores (
    context_key TEXT NOT NULL,
    row_id TEXT NOT NULL,
    cumulative_score REAL NOT NULL DEFAULT 0.0,
    n_signals INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (context_key, row_id)
);
"""


def _context_key(context_tags: dict[str, list[str]]) -> str:
    """Sérialise un contexte axial en clé stable pour l'index."""
    # Tri pour stabilité ; clés non-axiales ignorées.
    from muses.schemas.tags import AXIS_NAMES
    parts = []
    for axis in AXIS_NAMES:
        values = sorted(context_tags.get(axis, []))
        parts.append(f"{axis}:{','.join(values)}")
    return "|".join(parts)


@dataclass
class ContextRowScore:
    context_key: str
    row_id: str
    cumulative_score: float
    n_signals: int


class OnlineLearner:
    """Apprentissage online v0 : agrégat des signaux par (contexte, row).

    Le pondérateur étage 2 ou le filtreur étage 4 peuvent consulter
    `score(context_tags, row_id)` pour ajuster leur scoring.
    """

    def __init__(self, db_path: Path):
        self.db_path = Path(db_path)
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            conn.executescript(_SCHEMA)

    def update_from_signal(self, signal: FeedbackSignal) -> float:
        """Met à jour le score (context, row) selon le signal. Renvoie le delta appliqué."""
        delta = SIGNAL_SCORE_DELTAS.get(signal.signal, 0.0)
        if delta == 0.0:
            return 0.0
        ctx_key = _context_key(signal.context_tags or {})

        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT cumulative_score, n_signals FROM context_row_scores "
                "WHERE context_key=? AND row_id=?",
                (ctx_key, signal.row_id),
            ).fetchone()
            if row:
                new_score = row[0] + delta
                new_n = row[1] + 1
            else:
                new_score = delta
                new_n = 1
            conn.execute(
                "INSERT OR REPLACE INTO context_row_scores "
                "(context_key, row_id, cumulative_score, n_signals) "
                "VALUES (?, ?, ?, ?)",
                (ctx_key, signal.row_id, new_score, new_n),
            )
            conn.commit()
        return delta

    def get_score(self, context_tags: dict[str, list[str]], row_id: str) -> float:
        """Score normalisé (cumulative / n_signals) pour (contexte, row).

        Renvoie 0.0 si jamais observé. Bornes typiques : [-1, 1] sous les
        signaux courants.
        """
        ctx_key = _context_key(context_tags)
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT cumulative_score, n_signals FROM context_row_scores "
                "WHERE context_key=? AND row_id=?",
                (ctx_key, row_id),
            ).fetchone()
        if not row or row[1] == 0:
            return 0.0
        return row[0] / row[1]
