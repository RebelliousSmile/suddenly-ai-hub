"""T15 — Bootstrap initial : peuple une cellule contextuelle prioritaire.

Cellule cible (à confirmer après inventaire élargi du corpus) :
- univers = medieval_fantastique
- situation = combat
- rapport_initial = hostile
- voix = solennel
- emotion_dominante = colere

Sources :
- data/test-dataset-rp.jsonl   → fragments (assistant messages)
- muses.mining.beats           → 6 beats curés (rows beat)
- muses.mining.entities        → ~11 entités lexicales curées (rows entity)
- (templates : non bootstrappés en M1, à curer en M2 ou plus tard)

Le script :
1. Produit les rows pour chaque niveau peuplé
2. Ingère chacune via le pipeline M0
3. Reporte succès / échecs par niveau

Usage :
    PYTHONPATH=. python scripts/bootstrap_initial_cell.py
"""

from __future__ import annotations

from pathlib import Path

from muses.ingestion.pipeline import TablePaths, ingest
from muses.mining.beats import build_beat_rows
from muses.mining.crawl_adapter import extract_fragments_from_rp_dataset
from muses.mining.entities import build_entity_rows
from muses.tables.embeddings import StubEncoder


REPO_ROOT = Path(__file__).resolve().parent.parent
SOURCE_JSONL = REPO_ROOT / "data" / "test-dataset-rp.jsonl"
CELL_DIR = REPO_ROOT / "tables" / "bootstrap_cell_medfan_combat_hostile_solennel_colere"

TARGET_CELL_TAGS = {
    "univers": ["medieval_fantastique"],
    "situation": ["combat"],
    "rapport_initial": ["hostile"],
    "voix": ["solennel"],
    "emotion_dominante": ["colere"],
}


def _ingest_all(name: str, row_dicts: list[dict], cell_dir: Path) -> tuple[int, int]:
    """Ingère une liste de row dicts dans une table dédiée. Renvoie (success, failed)."""
    cell_dir.mkdir(parents=True, exist_ok=True)
    paths = TablePaths.from_dir(cell_dir, table_name=name)
    encoder = StubEncoder(dim=16)

    success = 0
    failed = 0
    for row_dict in row_dicts:
        result = ingest(row_dict, paths, encoder=encoder, verify_signature=False)
        if result.success:
            success += 1
        else:
            failed += 1
            print(f"  [{name}] REJECTED ({result.stage_failed}): {result.errors[0][:120]}")
    return success, failed


def _wipe_cell_dir() -> None:
    """Reset complet de la cellule cible — bootstrap est idempotent."""
    if not CELL_DIR.exists():
        return
    for child in CELL_DIR.iterdir():
        if child.is_file():
            child.unlink()


def main() -> int:
    print(f"Bootstrap initial cell from {SOURCE_JSONL}")
    print(f"  → cell dir: {CELL_DIR}")

    if not SOURCE_JSONL.exists():
        print(f"  ERROR: source missing: {SOURCE_JSONL}")
        return 1

    _wipe_cell_dir()
    totals = {}

    # FRAGMENTS — extraits du corpus RP
    fragment_rows = extract_fragments_from_rp_dataset(
        SOURCE_JSONL,
        tags=TARGET_CELL_TAGS,
        source="bootstrap",
        anonymize=True,
        min_chars=30,
    )
    totals["fragments"] = _ingest_all("fragments", fragment_rows, CELL_DIR)

    # BEATS — curés depuis le dict BEAT_KEYWORDS
    beat_rows = build_beat_rows(tags=TARGET_CELL_TAGS, source="bootstrap")
    totals["beats"] = _ingest_all("beats", beat_rows, CELL_DIR)

    # ENTITIES — curées depuis le LEXICON
    entity_rows = build_entity_rows(tags=TARGET_CELL_TAGS, source="bootstrap")
    totals["entities"] = _ingest_all("entities", entity_rows, CELL_DIR)

    # TEMPLATES — non bootstrappés en M1 (à curer manuellement en M2+)

    print()
    print("Summary:")
    overall_failed = 0
    for name, (s, f) in totals.items():
        marker = "OK" if f == 0 else "WARN"
        print(f"  [{marker}] {name:10s} — {s} ingested, {f} rejected")
        overall_failed += f

    return 0 if overall_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
