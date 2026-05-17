"""T49 — Résumé de session (pipeline hybride extraction + génération).

v0 : extraction des beats de chaque scène (par keyword matching) puis
remplissage d'un template de résumé minimal. Le ton "compte-rendu JDR
narratif" demandé par #83 sera mieux servi par des templates plus
sophistiqués curés dans M5+ — ici on démontre le mécanisme.
"""

from __future__ import annotations

from muses.mining.beats import BEAT_CATALOG, classify_beat_keywords


_SUMMARY_TEMPLATE = (
    "Résumé de session ({n_scenes} scène(s), ~{total_chars} caractères).\n\n"
    "Beats joués : {beats_list}.\n\n"
    "{scene_lines}"
)


def generate_session_summary(
    scene_fragments_per_scene: list[list[str]],
) -> str:
    """Produit un résumé narratif en remplissant un template.

    v0 : assemble un compte-rendu mécanique. Le polish narratif vient des
    templates curés (à enrichir par axe situation/voix).
    """
    if not scene_fragments_per_scene:
        return "Session vide — aucun fragment à résumer."

    total_chars = sum(
        sum(len(f) for f in fragments)
        for fragments in scene_fragments_per_scene
    )
    all_beats: set[str] = set()
    scene_lines = []
    for i, fragments in enumerate(scene_fragments_per_scene, start=1):
        scene_beats: set[str] = set()
        for f in fragments:
            scene_beats.update(classify_beat_keywords(f))
        all_beats.update(scene_beats)
        if scene_beats:
            scene_lines.append(
                f"Scène {i} ({len(fragments)} fragments) : {', '.join(sorted(scene_beats))}."
            )
        else:
            scene_lines.append(
                f"Scène {i} ({len(fragments)} fragments) : aucun beat saillant détecté."
            )

    return _SUMMARY_TEMPLATE.format(
        n_scenes=len(scene_fragments_per_scene),
        total_chars=total_chars,
        beats_list=", ".join(sorted(all_beats)) if all_beats else "aucun",
        scene_lines="\n".join(scene_lines),
    )
