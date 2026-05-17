"""HTTP API du service Muses.

Voir architecture-tables-ml.md § ApiMVP et external/use-cases.md §4.1
(MusesClient interface).
"""

from muses.api.schemas import SuggestRequest, SuggestResponse, SuggestionItem
from muses.api.server import create_app

__all__ = ["create_app", "SuggestRequest", "SuggestResponse", "SuggestionItem"]
