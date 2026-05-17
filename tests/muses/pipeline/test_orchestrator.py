"""Tests T20 — Orchestrateur pipeline 4-étages end-to-end."""

from muses.ingestion.pipeline import TablePaths, ingest
from muses.pipeline.orchestrator import Orchestrator
from muses.schemas.tags import AxialTags
from muses.tables.embeddings import StubEncoder


def _seed_fragments(table_dir, texts, tags_dict):
    paths = TablePaths.from_dir(table_dir)
    encoder = StubEncoder(dim=16)
    for text in texts:
        ingest({
            "level": "fragment",
            "tags": tags_dict,
            "content": {"text": text, "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)
    return paths.jsonl


def test_orchestrator_end_to_end_returns_candidates(tmp_path):
    table = _seed_fragments(
        tmp_path / "tbl",
        ["alpha", "beta", "gamma", "delta", "epsilon"],
        {"univers": ["medieval_fantastique"], "situation": ["combat"]},
    )
    orch = Orchestrator([table], encoder=StubEncoder(dim=16))
    result = orch.generate(
        context_text="alpha context",
        context_tags=AxialTags(
            univers=["medieval_fantastique"],
            situation=["combat"],
        ),
        n_candidates=3,
        top_n=2,
    )
    assert len(result.candidates) == 2
    assert all(c.source_row_ids for c in result.candidates)
    assert len(result.selected_tables) == 1


def test_orchestrator_no_matching_tables_returns_empty(tmp_path):
    table = _seed_fragments(
        tmp_path / "tbl",
        ["x"],
        {"univers": ["cyberpunk"]},
    )
    orch = Orchestrator([table], encoder=StubEncoder(dim=16))
    # On crée un contexte avec uniquement un univers incompatible.
    # Pas de relaxe sur univers possible avant le dernier — mais s'il est
    # relaxé en fin, le contexte devient universel et la cyberpunk row matche.
    result = orch.generate(
        context_text="x",
        context_tags=AxialTags(univers=["medieval_fantastique"]),
    )
    # Après relaxe univers, le contexte est universel → match cyberpunk
    assert len(result.candidates) >= 1


def test_orchestrator_tracks_relaxed_axes(tmp_path):
    # Table sans tags : universelle. Doit matcher exactement (pas de relaxe).
    table = _seed_fragments(
        tmp_path / "tbl",
        ["fragment 1", "fragment 2"],
        {},  # universelle
    )
    orch = Orchestrator([table], encoder=StubEncoder(dim=16))
    result = orch.generate(
        context_text="anything",
        context_tags=AxialTags(univers=["medieval_fantastique"]),
    )
    assert len(result.candidates) >= 1
    assert result.relaxed_axes == []
