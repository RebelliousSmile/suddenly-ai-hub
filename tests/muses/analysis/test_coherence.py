"""Tests T46-T48 — Analyse de cohérence."""

from muses.analysis.coherence import analyze_scene_coherence, analyze_session_coherence


def test_empty_scene_no_issues():
    assert analyze_scene_coherence([]) == []


def test_single_fragment_no_issues():
    assert analyze_scene_coherence(["Il dégaine son épée."]) == []


def test_consistent_beats_no_issues():
    fragments = [
        "Il dégaine son épée.",
        "Il frappe du plat de la lame.",  # tous deux = hostilite
    ]
    assert analyze_scene_coherence(fragments) == []


def test_disjoint_beats_flagged():
    fragments = [
        "Il dégaine son épée.",  # hostilite
        "Il caresse doucement sa main.",  # tendresse
    ]
    issues = analyze_scene_coherence(fragments)
    assert len(issues) == 1
    assert issues[0].severity == "warn"
    assert issues[0].fragment_index == 1


def test_session_aggregates_scene_issues():
    scenes = [
        ["Il dégaine son épée.", "Il caresse doucement sa main."],  # 1 issue
        ["Il frappe.", "Il attaque."],  # 0 issue
    ]
    result = analyze_session_coherence(scenes)
    assert result["n_scenes"] == 2
    assert result["n_issues"] == 1
    assert result["issues"][0]["scene_index"] == 0
