"""Pipeline 4-étages de génération.

Voir aidd_docs/memory/architecture-tables-ml.md § Pipeline de génération.
Versions v0 du MVP M2 — les versions v2 ML appris arrivent en M5/T51.
"""

from muses.pipeline.filter import NoOpFilter
from muses.pipeline.orchestrator import Candidate, GenerationResult, Orchestrator
from muses.pipeline.recombiner import FragmentPassthroughRecombiner
from muses.pipeline.selector import TableSelection, TagMatchingSelector
from muses.pipeline.weighter import CosineWeighter, WeightedRow

__all__ = [
    "Candidate",
    "CosineWeighter",
    "FragmentPassthroughRecombiner",
    "GenerationResult",
    "NoOpFilter",
    "Orchestrator",
    "TableSelection",
    "TagMatchingSelector",
    "WeightedRow",
]
