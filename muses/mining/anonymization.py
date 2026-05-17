"""Adapter d'anonymisation produisant du content row-compatible.

T11 du technical-plan. Si spaCy est disponible, on utilise le pipeline
`pipelines.anonymization.anonymize` existant. Sinon, fallback regex
conservateur (capitalisation + listes blanches d'exceptions courantes).

Sortie : texte avec noms propres remplacés par des placeholders typés
`{char.name}` (personnages) ou `{place.name}` (lieux).
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


# Mots fréquents capitalisés en début de phrase qu'on ne doit PAS confondre
# avec des noms propres dans le fallback regex.
_COMMON_CAPITALIZED = frozenset({
    "Je", "Tu", "Il", "Elle", "On", "Nous", "Vous", "Ils", "Elles",
    "Le", "La", "Les", "Un", "Une", "Des", "Du", "De", "Au", "Aux",
    "Mon", "Ma", "Mes", "Ton", "Ta", "Tes", "Son", "Sa", "Ses",
    "Notre", "Votre", "Leur", "Leurs",
    "Ce", "Cet", "Cette", "Ces",
    "Mais", "Et", "Ou", "Donc", "Or", "Ni", "Car",
    "Si", "Quand", "Comme", "Que", "Qui", "Quoi", "Où",
    "Oui", "Non", "Peut", "Pas",
    "Bonsoir", "Bonjour", "Bonne", "Bon",
    "Monsieur", "Madame", "Mademoiselle",
    "Seigneur", "Dame",
})

# Regex conservateur : un mot capitalisé qui n'est pas en début de phrase.
# Capture les séquences "Aldric", "Saint-Pierre", "Jeanne d'Arc" approximativement.
_NAME_PATTERN = re.compile(
    r"(?<![.!?]\s)(?<![.!?])(?<!^)"  # pas en début de phrase
    r"\b([A-ZÀ-Ý][a-zà-ÿ]+(?:[ '-][A-ZÀ-Ý][a-zà-ÿ]+)*)\b"
)


@dataclass
class AnonymizationResult:
    """Résultat d'une passe d'anonymisation."""

    text: str
    replacements: list[tuple[str, str]] = field(default_factory=list)
    backend: str = "regex"  # "spacy" | "regex"


def _spacy_available() -> bool:
    try:
        import spacy  # noqa: F401
        return True
    except ImportError:
        return False


def anonymize_text(text: str, *, force_backend: str | None = None) -> AnonymizationResult:
    """Anonymise un texte. Remplace noms propres par `{char.name}`.

    `force_backend` permet aux tests de désactiver explicitement spaCy
    même s'il est installé (`force_backend="regex"`).
    """
    backend = force_backend or ("spacy" if _spacy_available() else "regex")

    if backend == "spacy":
        return _anonymize_with_spacy(text)
    return _anonymize_with_regex(text)


def _anonymize_with_spacy(text: str) -> AnonymizationResult:
    """Délègue au pipeline existant `pipelines.anonymization.anonymize`."""
    from pipelines.anonymization.anonymize import anonymize_session

    # Le pipeline existant prend une "session" (liste de messages).
    # On wrappe le texte dans une session minimale.
    session = [{"role": "user", "content": text}]
    anonymized_session = anonymize_session(session)
    out_text = anonymized_session[0]["content"]
    # Le pipeline existant ne nous renvoie pas la liste des remplacements détaillés ;
    # on la laisse vide. Pour M1 c'est suffisant — le traçage fin viendra plus tard.
    return AnonymizationResult(text=out_text, replacements=[], backend="spacy")


def _anonymize_with_regex(text: str) -> AnonymizationResult:
    """Fallback regex sans dépendance externe.

    Stratégie conservatrice : remplace tout mot capitalisé hors début de
    phrase qui n'est pas dans la liste blanche. Risque de faux négatifs
    (noms en début de phrase non remplacés) mais peu de faux positifs.
    """
    replacements: list[tuple[str, str]] = []

    def _sub(match: re.Match) -> str:
        original = match.group(1)
        head = original.split()[0]
        if head in _COMMON_CAPITALIZED:
            return original
        replacements.append((original, "{char.name}"))
        return "{char.name}"

    out_text = _NAME_PATTERN.sub(_sub, text)
    return AnonymizationResult(text=out_text, replacements=replacements, backend="regex")
