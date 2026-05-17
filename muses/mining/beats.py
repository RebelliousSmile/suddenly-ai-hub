"""Classification heuristique des beats narratifs + production de beat rows.

T14 du technical-plan. v0 = keyword matching. La v2 ML (classifieur léger
entraîné sur exemples labellisés) viendra en M3/T30 ou plus tard.

Set initial de beats inspiré de la littérature narratologique. Délibérément
restreint à 6 beats — assez pour démontrer le mécanisme sans noyer un
classifieur naïf.
"""

from __future__ import annotations

from typing import Any


from typing import TypedDict


class BeatSpec(TypedDict):
    description: str
    keywords: list[str]
    arc_position: list[str]


# Source unique pour les beats curés v0. Tout couplage entre keywords,
# description et arc_position vit ici pour éviter les désynchronisations.
BEAT_CATALOG: dict[str, BeatSpec] = {
    "hesitation": {
        "description": "Le personnage hésite avant d'agir ou de parler.",
        "keywords": [
            "hésite", "hésitation", "incertain", "doute", "se demande",
            "ne sait pas", "indécis",
        ],
        "arc_position": ["debut", "milieu"],
    },
    "provocation": {
        "description": "Le personnage provoque l'autre, le pousse à réagir.",
        "keywords": [
            "défie", "défi", "provoque", "tu rigoles", "voyons voir",
            "voilà tout ce que", "ose", "viens donc",
        ],
        "arc_position": ["debut", "tournant"],
    },
    "menace": {
        "description": "Le personnage formule une menace explicite ou voilée.",
        "keywords": [
            "regretter", "menace", "tuer", "détruire", "anéantir",
            "tu vas mourir", "dernier souffle",
        ],
        "arc_position": ["milieu", "tournant"],
    },
    "revelation": {
        "description": "Une information importante est révélée.",
        "keywords": [
            "vérité", "révèle", "confesse", "avoue", "secret",
            "je dois te dire", "tu dois savoir",
        ],
        "arc_position": ["tournant", "fin"],
    },
    "hostilite": {
        "description": "Action physique hostile, dégainer ou frapper.",
        "keywords": [
            "lame", "épée", "couteau", "frappe", "attaque",
            "garde", "parade", "esquive",
        ],
        "arc_position": ["milieu", "tournant"],
    },
    "tendresse": {
        "description": "Geste ou parole de douceur, marque de proximité.",
        "keywords": [
            "main", "caresse", "doucement", "tendrement", "regarde",
            "sourire", "complicité",
        ],
        "arc_position": ["debut", "fin"],
    },
}

# Compat publique : mappings dérivés du catalogue. Importer plutôt
# BEAT_CATALOG dans le code nouveau.
BEAT_KEYWORDS: dict[str, list[str]] = {
    beat: spec["keywords"] for beat, spec in BEAT_CATALOG.items()
}


def classify_beat_keywords(text: str) -> list[str]:
    """Renvoie la liste des beats détectés par matching de mots-clés.

    Heuristique simple : un beat est candidat si au moins un de ses
    mots-clés apparaît (insensible à la casse) dans le texte. Renvoyés
    dans l'ordre du dict.
    """
    lower = text.lower()
    return [
        beat for beat, keywords in BEAT_KEYWORDS.items()
        if any(kw in lower for kw in keywords)
    ]


def build_beat_rows(
    *,
    tags: dict[str, list[str]],
    source: str = "bootstrap",
) -> list[dict[str, Any]]:
    """Produit la liste des row dicts (niveau `beat`) depuis BEAT_CATALOG.

    Une row par beat connu. Les `tags` axiaux sont appliqués uniformément
    (le caller filtre en amont selon la cellule cible).
    """
    rows: list[dict[str, Any]] = []
    for label, spec in BEAT_CATALOG.items():
        rows.append({
            "level": "beat",
            "tags": tags,
            "content": {
                "label": label,
                "description": spec["description"],
                "typical_templates": [],
                "arc_position": spec["arc_position"],
            },
            "source": source,
            "user_id": None,
            "instance_id": None,
            "signature": None,
        })
    return rows
