"""T26 — Event log des 5 signaux UI.

Append-only sur disque (JSONL). Source de vérité pour l'online learning,
le trust et l'audit. Voir style-coaching.md §3 pour la sémantique des
signaux et learning-and-trust.md §3 pour leur consommation.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Literal


SignalType = Literal[
    "accept",
    "accept_edited",
    "reject_off",
    "reject_challenge_appreciated",
    "ignore",
]


SIGNAL_TYPES: tuple[SignalType, ...] = (
    "accept",
    "accept_edited",
    "reject_off",
    "reject_challenge_appreciated",
    "ignore",
)


@dataclass
class FeedbackSignal:
    """Un signal UI émis par un utilisateur recevant une suggestion."""

    signal: SignalType
    # Contexte de la requête originale
    user_id: str             # ActivityPub URI du récepteur (auteur qui écrit)
    instance_id: str         # Instance source du récepteur
    feature: str             # ex: "dialogue"
    # Suggestion concernée
    row_id: str              # ID de la row qui a produit la suggestion
    contributor_user_id: str | None = None   # Auteur de la row (peut être null pour bootstrap)
    contributor_instance_id: str | None = None
    # Contexte axial au moment de la requête
    context_tags: dict[str, list[str]] = field(default_factory=dict)
    # Édition éventuelle (présente uniquement pour accept_edited)
    edited_text: str | None = None
    # Horodatage
    timestamp: datetime = field(default_factory=lambda: datetime.now(tz=timezone.utc))

    def to_dict(self) -> dict:
        d = asdict(self)
        d["timestamp"] = self.timestamp.isoformat()
        return d

    @classmethod
    def from_dict(cls, data: dict) -> "FeedbackSignal":
        d = dict(data)
        d["timestamp"] = datetime.fromisoformat(d["timestamp"])
        return cls(**d)


class EventLog:
    """Log append-only des signaux UI, persisté en JSONL."""

    def __init__(self, path: Path):
        self.path = Path(path)

    def append(self, signal: FeedbackSignal) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(signal.to_dict(), ensure_ascii=False) + "\n")

    def iter_signals(self) -> Iterator[FeedbackSignal]:
        if not self.path.exists():
            return
        with self.path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield FeedbackSignal.from_dict(json.loads(line))

    def count(self) -> int:
        if not self.path.exists():
            return 0
        with self.path.open("r", encoding="utf-8") as fh:
            return sum(1 for line in fh if line.strip())
