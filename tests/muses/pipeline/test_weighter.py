"""Tests Étage 2 — Pondérateur cosinus."""

import numpy as np

from muses.ingestion.pipeline import TablePaths, ingest
from muses.pipeline.weighter import CosineWeighter, _cosine_similarity
from muses.schemas.tags import AxialTags
from muses.tables.embeddings import StubEncoder


def test_cosine_similarity_perfect_match():
    a = np.array([1.0, 0.0, 0.0], dtype=np.float32)
    b = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32)
    scores = _cosine_similarity(a, b)
    assert scores[0] == 1.0
    assert scores[1] == 0.0
    assert scores[2] == -1.0


def test_cosine_handles_zero_norm():
    a = np.zeros(3, dtype=np.float32)
    b = np.array([[1.0, 0.0, 0.0]], dtype=np.float32)
    scores = _cosine_similarity(a, b)
    assert scores[0] == 0.0


def _seed(table_dir, texts, tags):
    paths = TablePaths.from_dir(table_dir)
    encoder = StubEncoder(dim=16)
    for text in texts:
        ingest({
            "level": "fragment",
            "tags": tags,
            "content": {"text": text, "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)
    return paths.jsonl


def test_rank_orders_by_descending_score(tmp_path):
    tags = {"univers": ["medieval_fantastique"]}
    table = _seed(tmp_path, ["alpha", "beta", "gamma"], tags)
    encoder = StubEncoder(dim=16)
    weighter = CosineWeighter(encoder)
    weighted = weighter.rank([table], context_text="alpha")
    assert len(weighted) == 3
    # Décroissant
    scores = [w.score for w in weighted]
    assert scores == sorted(scores, reverse=True)


def test_rank_top_k_limit(tmp_path):
    tags = {"univers": ["medieval_fantastique"]}
    table = _seed(tmp_path, [f"row {i}" for i in range(5)], tags)
    encoder = StubEncoder(dim=16)
    weighter = CosineWeighter(encoder)
    weighted = weighter.rank([table], context_text="anything", top_k=2)
    assert len(weighted) == 2


def test_rank_filters_by_context_tags(tmp_path):
    paths_med = TablePaths.from_dir(tmp_path / "med")
    paths_cyber = TablePaths.from_dir(tmp_path / "cyber")
    encoder = StubEncoder(dim=16)
    for d, txt, univers in [
        (paths_med, "medieval text", "medieval_fantastique"),
        (paths_cyber, "cyber text", "cyberpunk"),
    ]:
        ingest({
            "level": "fragment",
            "tags": {"univers": [univers]},
            "content": {"text": txt, "char_pov": "neutral"},
            "source": "bootstrap",
        }, d, encoder=encoder, verify_signature=False)

    weighter = CosineWeighter(encoder)
    weighted = weighter.rank(
        [paths_med.jsonl, paths_cyber.jsonl],
        context_text="anything",
        context_tags=AxialTags(univers=["medieval_fantastique"]),
    )
    assert len(weighted) == 1
    assert "medieval" in weighted[0].row.parsed_content().text


def test_rank_empty_input(tmp_path):
    encoder = StubEncoder(dim=8)
    weighter = CosineWeighter(encoder)
    assert weighter.rank([], context_text="x") == []
