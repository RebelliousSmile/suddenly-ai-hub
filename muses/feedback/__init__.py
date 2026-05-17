"""Boucle de feedback Muses : signaux UI, trust, profil, garde-fous.

Voir aidd_docs/memory/learning-and-trust.md et style-coaching.md.
"""

from muses.feedback.events import SIGNAL_TYPES, EventLog, FeedbackSignal
from muses.feedback.guardrails import AntiSleeperGuard
from muses.feedback.instance_reputation import InstanceReputationStore
from muses.feedback.meta_suggestions import MetaSuggestionGenerator
from muses.feedback.online_learning import OnlineLearner
from muses.feedback.style_profile import StyleProfileStore
from muses.feedback.trust import BetaReputation, TrustStore

__all__ = [
    "SIGNAL_TYPES",
    "AntiSleeperGuard",
    "BetaReputation",
    "EventLog",
    "FeedbackSignal",
    "InstanceReputationStore",
    "MetaSuggestionGenerator",
    "OnlineLearner",
    "StyleProfileStore",
    "TrustStore",
]
