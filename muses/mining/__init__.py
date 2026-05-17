"""Pipeline de mining bootstrap : source brute → rows candidates.

Voir aidd_docs/memory/external/corpus-public.md § Du corpus brut aux rows
pour le flux. Cette implémentation est volontairement minimale (v0) :
- l'anonymisation utilise spaCy si disponible (extra `pipelines`), sinon
  remplace tout nom propre détecté par regex par un placeholder ;
- l'extraction d'entités et la segmentation en beats sont des stubs
  heuristiques destinés à être remplacés par des classifieurs entraînés en
  M3/T30 et au-delà.

Voir technical-plan.md M1 (T11-T15) et M5 (T51 pour les versions v2 ML).
"""

from muses.mining.anonymization import AnonymizationResult, anonymize_text
from muses.mining.beats import (
    BEAT_KEYWORDS,
    build_beat_rows,
    classify_beat_keywords,
)
from muses.mining.crawl_adapter import (
    extract_fragments_from_rp_dataset,
    parse_rp_dataset_jsonl,
)
from muses.mining.entities import LEXICON, build_entity_rows

__all__ = [
    "AnonymizationResult",
    "anonymize_text",
    "BEAT_KEYWORDS",
    "build_beat_rows",
    "classify_beat_keywords",
    "extract_fragments_from_rp_dataset",
    "parse_rp_dataset_jsonl",
    "LEXICON",
    "build_entity_rows",
]
