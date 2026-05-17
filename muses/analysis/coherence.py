"""T46-T48 — Analyse de cohérence scène/session.

Pour le MVP M5, version minimale : classification de chaque fragment d'une
scène contre le `BEAT_CATALOG` heuristique (cf. mining/beats.py) puis
détection de transitions abruptes (changement de beat sans transition
narrative attendue).

Une version réellement riche nécessite des tables de patterns d'incohérence
curées + un embedder fin — à itérer après le MVP. Le scaffold ici expose
l'API attendue côté instance Suddenly.
"""

from __future__ import annotations

from dataclasses import dataclass

from muses.mining.beats import BEAT_CATALOG, classify_beat_keywords


@dataclass
class CoherenceIssue:
    """Un problème de cohérence détecté."""

    severity: str   # "info" | "warn" | "error"
    fragment_index: int
    description: str


def analyze_scene_coherence(scene_fragments: list[str]) -> list[CoherenceIssue]:
    """Analyse simple : détecte les ruptures de beat consécutives.

    Une "rupture" v0 = deux fragments consécutifs dont les sets de beats
    sont entièrement disjoints. Heuristique grossière — sera affinée par
    un classifieur de transitions canoniques en M5+.
    """
    if not scene_fragments:
        return []

    beat_sequence = [set(classify_beat_keywords(f)) for f in scene_fragments]
    issues: list[CoherenceIssue] = []
    for i in range(1, len(beat_sequence)):
        prev = beat_sequence[i - 1]
        curr = beat_sequence[i]
        if prev and curr and prev.isdisjoint(curr):
            issues.append(CoherenceIssue(
                severity="warn",
                fragment_index=i,
                description=(
                    f"Rupture de beat entre fragments {i - 1} (beats: {sorted(prev)}) "
                    f"et {i} (beats: {sorted(curr)})."
                ),
            ))
    return issues


def analyze_session_coherence(scene_fragments_per_scene: list[list[str]]) -> dict:
    """Cohérence session = somme des analyses scène + détection d'arc global.

    v0 : juste l'agrégat des issues scène-par-scène plus le compte de beats
    distincts joués dans la session. Le scoring d'arc canonique est en
    M5+ (analyse-pipeline.md à figer).
    """
    all_issues: list[tuple[int, CoherenceIssue]] = []
    all_beats: set[str] = set()
    for scene_idx, fragments in enumerate(scene_fragments_per_scene):
        issues = analyze_scene_coherence(fragments)
        for issue in issues:
            all_issues.append((scene_idx, issue))
        for f in fragments:
            all_beats.update(classify_beat_keywords(f))

    return {
        "n_scenes": len(scene_fragments_per_scene),
        "n_issues": len(all_issues),
        "distinct_beats": sorted(all_beats),
        "issues": [
            {
                "scene_index": s_idx,
                "fragment_index": issue.fragment_index,
                "severity": issue.severity,
                "description": issue.description,
            }
            for s_idx, issue in all_issues
        ],
    }
