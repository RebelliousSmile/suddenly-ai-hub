"""Persistance CSV des votes du banc A/B."""

from __future__ import annotations

import csv
import os
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
VOTES_CSV = REPO_ROOT / "data" / "playground" / "votes.csv"

FIELDS = [
    "timestamp",
    "feature_adapter",
    "model_left",
    "model_right",
    "prompt",
    "response_left",
    "response_right",
    "vote",
    "note",
]


def append_vote(
    feature_adapter: str,
    model_left: str,
    model_right: str,
    prompt: str,
    response_left: str,
    response_right: str,
    vote: str,
    note: str = "",
) -> None:
    VOTES_CSV.parent.mkdir(parents=True, exist_ok=True)
    write_header = not VOTES_CSV.exists()
    with open(VOTES_CSV, "a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "feature_adapter": feature_adapter,
            "model_left": model_left,
            "model_right": model_right,
            "prompt": prompt,
            "response_left": response_left,
            "response_right": response_right,
            "vote": vote,
            "note": note,
        })


def read_recent(limit: int = 50) -> list[dict]:
    if not VOTES_CSV.exists():
        return []
    with open(VOTES_CSV, encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    return rows[-limit:][::-1]


def aggregate_scores() -> dict[str, dict[str, int]]:
    """Renvoie {model: {wins, losses, ties}} agrégé sur tout le CSV."""
    scores: dict[str, dict[str, int]] = {}
    for row in read_recent(limit=10_000):
        left = row["model_left"]
        right = row["model_right"]
        vote = row["vote"]
        for m in (left, right):
            scores.setdefault(m, {"wins": 0, "losses": 0, "ties": 0})
        if vote == "left":
            scores[left]["wins"] += 1
            scores[right]["losses"] += 1
        elif vote == "right":
            scores[right]["wins"] += 1
            scores[left]["losses"] += 1
        elif vote == "tie":
            scores[left]["ties"] += 1
            scores[right]["ties"] += 1
    return scores
