"""Extraction d'entités lexicales (v0 lexicon-based).

T13 du technical-plan. v0 = matching lexical sur un dictionnaire curé. Une
version v2 ferait du NER spaCy avec clustering de variantes morphologiques,
prévue en M3/T30 ou plus tard.

Le lexique initial est restreint et orienté `medieval_fantastique` + `combat`
pour le bootstrap de la cellule prioritaire. À élargir au fur et à mesure
que d'autres cellules sont couvertes.
"""

from __future__ import annotations

from typing import Any


# Lexique : type d'entité → dict {lemma: forms}
# forms : dict "<genre>.<nombre>.<temps>" → forme effective. Les clés sont
# en snake_case ASCII pour cohérence avec data-format.md.
LEXICON: dict[str, dict[str, dict[str, str]]] = {
    "geste": {
        "serrer les poings": {
            "m.s.present": "serre les poings",
            "f.s.present": "serre les poings",
            "m.p.present": "serrent les poings",
        },
        "degainer son epee": {
            "m.s.present": "dégaine son épée",
            "f.s.present": "dégaine son épée",
        },
        "lever la garde": {
            "m.s.present": "lève la garde",
            "f.s.present": "lève la garde",
        },
        "frapper du plat de la lame": {
            "m.s.present": "frappe du plat de la lame",
        },
    },
    "emotion": {
        "colere froide": {"_": "colère froide"},
        "rage muette": {"_": "rage muette"},
        "indignation": {"_": "indignation"},
        "mepris": {"_": "mépris"},
    },
    "lieu": {
        "salle d'armes": {"_": "salle d'armes"},
        "cour interieure": {"_": "cour intérieure"},
        "garnison": {"_": "garnison"},
    },
}


def build_entity_rows(
    *,
    tags: dict[str, list[str]],
    source: str = "bootstrap",
    lexicon: dict | None = None,
) -> list[dict[str, Any]]:
    """Produit la liste des row dicts (niveau `entity`) depuis le lexique.

    Chaque entrée du lexique devient une row. Le caller fournit les `tags`
    axiaux à appliquer.
    """
    lex = lexicon or LEXICON
    rows: list[dict[str, Any]] = []
    for ent_type, entries in lex.items():
        for lemma, forms in entries.items():
            rows.append({
                "level": "entity",
                "tags": tags,
                "content": {
                    "type": ent_type,
                    "lemma": lemma,
                    "variants": _infer_variants(forms),
                    "forms": forms,
                },
                "source": source,
                "user_id": None,
                "instance_id": None,
                "signature": None,
            })
    return rows


def _infer_variants(forms: dict[str, str]) -> dict[str, list[str]]:
    """Reconstruit les axes de variation depuis les clés des formes."""
    axes: dict[str, set[str]] = {}
    for key in forms:
        if key == "_":
            continue
        parts = key.split(".")
        for i, part in enumerate(parts):
            axis_name = ["genre", "nombre", "tense"][i] if i < 3 else f"axis_{i}"
            axes.setdefault(axis_name, set()).add(part)
    return {axis: sorted(values) for axis, values in axes.items()}
