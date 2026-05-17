"""Snapshot manuel des stores feedback. Utilisé par cron pour T40.

Usage :
    PYTHONPATH=. python scripts/snapshot_feedback.py

Lit `MUSES_FEEDBACK_DIR` et `MUSES_SNAPSHOT_DIR` depuis l'environnement
(cf. .env.example). Crée un sous-dossier horodaté dans
`$MUSES_SNAPSHOT_DIR/` avec une copie de `$MUSES_FEEDBACK_DIR/`.

Exemple de cron (snapshot horaire + nettoyage des snapshots > 7 jours) :

    0 * * * * cd /opt/muses && PYTHONPATH=. python scripts/snapshot_feedback.py
    0 3 * * * find /opt/muses/snapshots/ -maxdepth 1 -type d -mtime +7 -exec rm -rf {} +
"""

from __future__ import annotations

import os
from pathlib import Path

from muses.feedback.snapshots import snapshot_directory


def main() -> int:
    feedback_dir = Path(os.environ.get("MUSES_FEEDBACK_DIR", "feedback"))
    snapshot_dir = Path(os.environ.get("MUSES_SNAPSHOT_DIR", "snapshots"))

    if not feedback_dir.exists():
        print(f"ERROR: feedback dir absent: {feedback_dir}")
        return 1

    snap = snapshot_directory(feedback_dir, snapshot_dir)
    print(f"snapshot OK → {snap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
