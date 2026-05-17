"""T40 — Snapshots et rollback des stores ML / feedback.

Voir learning-and-trust.md § Snapshots et rollback. Pour le MVP, les
"poids ML" vivent dans les SQLite stores du module `feedback/` (trust,
instance_reputation, style_profile, online_learning). Un snapshot est une
copie horodatée de ces fichiers.

Le rollback ne supprime PAS les rows ajoutées entre-temps — seuls les
poids dérivés sont rétablis. Les rows JSONL restent intactes.
"""

from __future__ import annotations

import shutil
from datetime import datetime, timezone
from pathlib import Path


SNAPSHOT_TS_FORMAT = "%Y%m%dT%H%M%SZ"


def _ts_now() -> str:
    return datetime.now(tz=timezone.utc).strftime(SNAPSHOT_TS_FORMAT)


def snapshot_directory(source_dir: Path, snapshot_dir: Path) -> Path:
    """Copie le contenu de `source_dir` dans `snapshot_dir/<ts>/`.

    Renvoie le chemin du sous-dossier créé.
    """
    source_dir = Path(source_dir)
    snapshot_dir = Path(snapshot_dir)
    snapshot_dir.mkdir(parents=True, exist_ok=True)
    target = snapshot_dir / _ts_now()
    if target.exists():
        # collision improbable (résolution seconde), on suffixe
        target = snapshot_dir / f"{_ts_now()}_{id(target) & 0xFFFF:04x}"
    shutil.copytree(source_dir, target)
    return target


def list_snapshots(snapshot_dir: Path) -> list[Path]:
    """Renvoie les snapshots disponibles, ordre antichronologique."""
    snapshot_dir = Path(snapshot_dir)
    if not snapshot_dir.exists():
        return []
    children = [c for c in snapshot_dir.iterdir() if c.is_dir()]
    children.sort(key=lambda p: p.name, reverse=True)
    return children


def restore_snapshot(snapshot_path: Path, target_dir: Path) -> None:
    """Restaure un snapshot vers `target_dir` (écrase le contenu actuel)."""
    snapshot_path = Path(snapshot_path)
    target_dir = Path(target_dir)
    if not snapshot_path.exists():
        raise FileNotFoundError(f"Snapshot inexistant: {snapshot_path}")
    if target_dir.exists():
        shutil.rmtree(target_dir)
    shutil.copytree(snapshot_path, target_dir)
