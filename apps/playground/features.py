"""Catalogue des features Suddenly AI (source : aidd_docs/memory/external/use-cases.md)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Feature:
    issue: int
    name: str
    adapter: str
    muses: str
    kind: str
    system_prompt: str
    sample_user_prompt: str


FEATURES: list[Feature] = [
    Feature(
        issue=77,
        name="Suggestion de dialogue",
        adapter="suddenly-dialogue",
        muses="1-3",
        kind="Génération",
        system_prompt=(
            "Tu es un assistant d'écriture RP. Produis une réplique de dialogue à la "
            "première personne, cohérente avec le personnage et la scène. Pas de "
            "didascalies, pas de méta-commentaire."
        ),
        sample_user_prompt=(
            "Personnage : Aelis, mage elfe novice, hésitante mais curieuse.\n"
            "Scène : la bibliothèque interdite, minuit. Elle vient d'ouvrir un grimoire "
            "qui parle.\n"
            "Tâche : suggère la prochaine réplique d'Aelis."
        ),
    ),
    Feature(
        issue=78,
        name="Suggestion d'action",
        adapter="suddenly-action",
        muses="1-3",
        kind="Génération",
        system_prompt=(
            "Tu es un assistant d'écriture RP. Décris une action concrète cohérente avec "
            "le personnage et la situation. Style narratif, troisième personne, présent."
        ),
        sample_user_prompt=(
            "Personnage : Kael, garde frontalier, méfiant, blessé à l'épaule.\n"
            "Scène : embuscade au col de Tassaran. Trois silhouettes encapuchonnées "
            "viennent d'apparaître.\n"
            "Tâche : suggère l'action immédiate de Kael."
        ),
    ),
    Feature(
        issue=79,
        name="Suggestion de description",
        adapter="suddenly-description",
        muses="3",
        kind="Génération",
        system_prompt=(
            "Tu es un assistant d'écriture RP. Produis une description sensorielle "
            "d'ambiance ou de lieu, style narratif, troisième personne."
        ),
        sample_user_prompt=(
            "Lieu : taverne du Cheval Bleu, soir d'orage, salle pleine.\n"
            "Tâche : décris l'ambiance perçue à l'entrée."
        ),
    ),
    Feature(
        issue=80,
        name="Pensée intérieure",
        adapter="suddenly-thought",
        muses="5",
        kind="Génération",
        system_prompt=(
            "Tu es un assistant d'écriture RP. Produis un monologue intérieur à la "
            "première personne, qui révèle le ressenti et les arrière-pensées du "
            "personnage. Pas d'action visible."
        ),
        sample_user_prompt=(
            "Personnage : Mira, diplomate, masque la peur en public.\n"
            "Scène : audience au palais, l'ambassadeur ennemi vient d'entrer.\n"
            "Tâche : la pensée intérieure de Mira."
        ),
    ),
    Feature(
        issue=81,
        name="Analyse cohérence scène",
        adapter="suddenly-consistency-scene",
        muses="5",
        kind="Analyse",
        system_prompt=(
            "Tu es un analyste narratif RP. Repère les incohérences de caractère ou de "
            "ton dans la scène fournie. Liste-les en bullets brefs avec l'extrait fautif."
        ),
        sample_user_prompt=(
            "[Scène]\n"
            "Kael (taciturne dans les chapitres précédents) : \"Oh la la, quelle "
            "magnifique journée pour aller cueillir des marguerites !\"\n"
            "Mira (mage de combat) tire son épée à deux mains.\n"
            "Le ciel devient rose puis vert.\n\n"
            "Tâche : analyse les incohérences."
        ),
    ),
    Feature(
        issue=82,
        name="Analyse cohérence session",
        adapter="suddenly-consistency-session",
        muses="10",
        kind="Analyse",
        system_prompt=(
            "Tu es un analyste narratif RP. Résume l'arc de chaque personnage de la "
            "session et signale les ruptures."
        ),
        sample_user_prompt=(
            "[Session — 3 scènes]\n"
            "Scène 1 : Aelis découvre le grimoire parlant.\n"
            "Scène 2 : Kael défend le col de Tassaran.\n"
            "Scène 3 : Mira et Aelis confrontent l'ambassadeur, Kael trahit le groupe.\n\n"
            "Tâche : analyse les arcs et incohérences."
        ),
    ),
    Feature(
        issue=83,
        name="Résumé de session",
        adapter="suddenly-summary",
        muses="10",
        kind="Génération",
        system_prompt=(
            "Tu es un chroniqueur RP. Produis un résumé narratif de la session, "
            "troisième personne, ~500 mots, style compte-rendu JDR."
        ),
        sample_user_prompt=(
            "Session : Aelis, Kael, Mira au col de Tassaran. Découverte d'un grimoire "
            "parlant, embuscade, trahison de Kael, fuite vers la cité-état d'Halad.\n\n"
            "Tâche : résumé."
        ),
    ),
    Feature(
        issue=84,
        name="Suggestions de liens fédérés",
        adapter="suddenly-federation",
        muses="20",
        kind="Analyse",
        system_prompt=(
            "Tu es un assistant de fédération RP. Compare les PNJ de la session avec "
            "les personnages publics fournis et propose claim/adopt/fork avec un score "
            "de pertinence (fort/moyen/faible)."
        ),
        sample_user_prompt=(
            "PNJ de la session : Sire Halmar (vieux chevalier, exil volontaire, parle "
            "peu).\n"
            "Personnages publics :\n"
            " - Halmar le Taciturne (chevalier exilé, instance fantasy.suddenly.social)\n"
            " - Garrek Brisefer (mercenaire bavard, fantasy.suddenly.social)\n\n"
            "Tâche : propose les liens."
        ),
    ),
    Feature(
        issue=89,
        name="Export prompt vidéo",
        adapter="suddenly-description (étendu)",
        muses="5",
        kind="Génération",
        system_prompt=(
            "Tu es un assistant de pré-production vidéo. Produis un prompt structuré "
            "pour un générateur vidéo (Sora/Runway/Kling) : lieu, ambiance, "
            "personnages, action, cadrage."
        ),
        sample_user_prompt=(
            "Scène : duel au sabre dans une cour pavée sous la pluie, deux duellistes, "
            "spectateurs en arrière-plan, lumière de torche.\n\n"
            "Tâche : prompt vidéo."
        ),
    ),
]


def feature_by_adapter(adapter: str) -> Feature | None:
    for f in FEATURES:
        if f.adapter == adapter:
            return f
    return None
