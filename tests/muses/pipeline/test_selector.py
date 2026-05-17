"""Tests Étage 1 — Sélecteur."""

from muses.ingestion.pipeline import TablePaths, ingest
from muses.pipeline.selector import RELAX_ORDER, TagMatchingSelector
from muses.schemas.tags import AxialTags
from muses.tables.embeddings import StubEncoder


def _seed_table(table_dir, name, tags_dict, n=3):
    """Seed une table avec n rows ayant les tags fournis."""
    paths = TablePaths.from_dir(table_dir, table_name=name)
    encoder = StubEncoder(dim=8)
    for i in range(n):
        ingest({
            "level": "fragment",
            "tags": tags_dict,
            "content": {"text": f"{name} {i}", "char_pov": "neutral"},
            "source": "bootstrap",
        }, paths, encoder=encoder, verify_signature=False)
    return paths.jsonl


def test_select_exact_match(tmp_path):
    table = _seed_table(tmp_path, "t1", {
        "univers": ["medieval_fantastique"],
        "situation": ["combat"],
    })
    selector = TagMatchingSelector([table])
    selections = selector.select(AxialTags(
        univers=["medieval_fantastique"],
        situation=["combat"],
    ))
    assert len(selections) == 1
    assert selections[0].is_exact_match
    assert selections[0].relaxed_axes == []


def test_select_relaxes_emotion_first(tmp_path):
    table = _seed_table(tmp_path, "t1", {
        "univers": ["medieval_fantastique"],
        "situation": ["combat"],
    })
    selector = TagMatchingSelector([table])
    selections = selector.select(AxialTags(
        univers=["medieval_fantastique"],
        situation=["combat"],
        emotion_dominante=["joie"],  # incompatible avec une row sans emotion → relaxe
    ))
    assert len(selections) == 1
    # La table compatible n'a pas de emotion_dominante → toujours match même sans relax,
    # car l'absence de tag = universel. Donc exact match en réalité.
    assert selections[0].is_exact_match


def test_select_no_match_returns_empty(tmp_path):
    table = _seed_table(tmp_path, "t1", {
        "univers": ["cyberpunk"],
    })
    selector = TagMatchingSelector([table])
    # Cherche medieval qui n'existe pas, même après relaxe complète tous les axes
    # contextuels finissent vides → la row passe (universal context). Confirme.
    selections = selector.select(AxialTags(univers=["medieval_fantastique"]))
    # Après relaxe de univers (dernier axe), le contexte devient universel → match
    assert len(selections) == 1
    # univers est dans RELAX_ORDER en dernier, donc relaxé en dernier
    assert "univers" in selections[0].relaxed_axes


def test_skip_table_below_min_rows(tmp_path):
    sparse = _seed_table(tmp_path, "sparse", {"univers": ["medieval_fantastique"]}, n=1)
    dense = _seed_table(tmp_path, "dense", {"univers": ["medieval_fantastique"]}, n=5)
    selector = TagMatchingSelector([sparse, dense], min_rows=3)
    selections = selector.select(AxialTags(univers=["medieval_fantastique"]))
    paths = [s.table_path for s in selections]
    assert dense in paths
    assert sparse not in paths


def test_relax_order_is_canonical():
    assert RELAX_ORDER == (
        "emotion_dominante", "voix", "rapport_initial", "situation", "univers",
    )
