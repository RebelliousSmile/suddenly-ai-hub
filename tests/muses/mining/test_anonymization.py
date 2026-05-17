"""Tests de l'adapter d'anonymisation (fallback regex)."""

from muses.mining.anonymization import anonymize_text


class TestRegexFallback:
    def test_substitutes_proper_name(self):
        result = anonymize_text(
            "Le marchand regarde Aldric avec méfiance.", force_backend="regex"
        )
        assert "Aldric" not in result.text
        assert "{char.name}" in result.text
        assert ("Aldric", "{char.name}") in result.replacements

    def test_keeps_sentence_initial_capitalized(self):
        """Le mot capitalisé en début de phrase n'est pas substitué (limitation v0)."""
        result = anonymize_text("Aldric entre dans la taverne.", force_backend="regex")
        # On documente la limitation : début de phrase non traité par le regex.
        assert "Aldric" in result.text

    def test_keeps_common_words(self):
        result = anonymize_text(
            "Le marchand dit : 'Bonjour, Monsieur.'", force_backend="regex"
        )
        assert "Monsieur" in result.text
        assert "Bonjour" in result.text
        assert "{char.name}" not in result.text

    def test_handles_multi_word_name(self):
        result = anonymize_text(
            "Le chevalier salue Jean-Paul Dupont avec respect.", force_backend="regex"
        )
        assert "{char.name}" in result.text
        assert "Jean-Paul Dupont" not in result.text

    def test_no_replacement_yields_empty_list(self):
        result = anonymize_text("Le ciel est gris.", force_backend="regex")
        assert result.replacements == []

    def test_backend_reported(self):
        result = anonymize_text("test", force_backend="regex")
        assert result.backend == "regex"
