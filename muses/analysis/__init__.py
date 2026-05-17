"""Pipeline d'analyse par projection inversée.

Voir architecture-tables-ml.md § Pipeline d'analyse — projection inversée
et external/use-cases.md §3 pour les 4 features d'analyse.

Module créé en M5/T45. Versions complètes des features individuelles
viennent par incréments dans cette même phase.
"""

from muses.analysis.coherence import analyze_scene_coherence, analyze_session_coherence
from muses.analysis.federated_links import find_federated_links
from muses.analysis.matcher import EmbeddingMatcher, MatchResult
from muses.analysis.summary import generate_session_summary

__all__ = [
    "analyze_scene_coherence",
    "analyze_session_coherence",
    "EmbeddingMatcher",
    "find_federated_links",
    "generate_session_summary",
    "MatchResult",
]
