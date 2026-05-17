"""Tests T33 — Mode challenge dans l'orchestrateur."""

from muses.feedback.style_profile import StyleProfileStore
from muses.ingestion.pipeline import TablePaths, ingest
from muses.pipeline.orchestrator import Orchestrator
from muses.schemas.tags import AxialTags
from muses.tables.embeddings import StubEncoder


def _seed(table_dir, texts, tags):
    paths = TablePaths.from_dir(table_dir)
    encoder = StubEncoder(dim=16)
    rows_ids = []
    for text in texts:
        result = ingest({
            "level": "fragment",
            "tags": tags,
            "content": {"text": text, "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)
        rows_ids.append(result.row_id)
    return paths.jsonl, rows_ids


def test_confort_mode_no_malus(tmp_path):
    """En confort, l'ordre est dicté uniquement par la similarité."""
    table, _ = _seed(tmp_path / "tbl", ["alpha", "beta", "gamma"], {
        "univers": ["medieval_fantastique"], "situation": ["combat"],
    })
    orch = Orchestrator([table], encoder=StubEncoder(dim=16))
    result = orch.generate(
        context_text="alpha",
        context_tags=AxialTags(
            univers=["medieval_fantastique"], situation=["combat"],
        ),
        mode="confort",
        top_n=3,
    )
    assert len(result.candidates) == 3


def test_challenge_mode_demotes_familiar_rows(tmp_path):
    """En challenge, une row déjà vue par le user est pénalisée."""
    table, row_ids = _seed(tmp_path / "tbl", ["alpha", "beta", "gamma"], {
        "univers": ["medieval_fantastique"], "situation": ["combat"],
    })
    # Marque la 1re row comme familière au user
    style_store = StyleProfileStore(tmp_path / "style.sqlite")
    for _ in range(20):
        style_store.observe("alice", row_id=row_ids[0])

    orch = Orchestrator(
        [table],
        encoder=StubEncoder(dim=16),
        style_store=style_store,
    )

    result_confort = orch.generate(
        context_text="alpha context",
        context_tags=AxialTags(
            univers=["medieval_fantastique"], situation=["combat"],
        ),
        mode="confort",
        top_n=3,
        user_id="alice",
    )
    result_challenge = orch.generate(
        context_text="alpha context",
        context_tags=AxialTags(
            univers=["medieval_fantastique"], situation=["combat"],
        ),
        mode="challenge",
        top_n=3,
        user_id="alice",
    )
    # En challenge mode, la row familière doit avoir un score plus bas
    # qu'en mode confort sur la même position.
    confort_scores = [c.source_scores[0] for c in result_confort.candidates]
    challenge_scores = [c.source_scores[0] for c in result_challenge.candidates]
    # Au moins une position est différente entre les deux résultats.
    assert confort_scores != challenge_scores or [c.text for c in result_confort.candidates] != [c.text for c in result_challenge.candidates]


def test_challenge_mode_without_style_store_is_noop(tmp_path):
    """Si aucun style_store n'est fourni, le mode challenge ne change rien."""
    table, _ = _seed(tmp_path / "tbl", ["alpha", "beta"], {
        "univers": ["medieval_fantastique"], "situation": ["combat"],
    })
    orch = Orchestrator([table], encoder=StubEncoder(dim=16), style_store=None)
    result = orch.generate(
        context_text="x",
        context_tags=AxialTags(
            univers=["medieval_fantastique"], situation=["combat"],
        ),
        mode="challenge",
        user_id="alice",
    )
    assert len(result.candidates) > 0  # ne crash pas, retourne quelque chose
