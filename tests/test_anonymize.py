from unittest.mock import patch, MagicMock

import pytest

from pipeline.anonymize import anonymize_session, _replace_persons, _detect_lang


def _make_ent(text: str, label: str, start_char: int, end_char: int) -> MagicMock:
    ent = MagicMock()
    ent.text = text
    ent.label_ = label
    ent.start_char = start_char
    ent.end_char = end_char
    return ent


def _make_doc(ents: list) -> MagicMock:
    doc = MagicMock()
    doc.ents = ents
    return doc


def _make_nlp(entities_by_text: dict[str, list[tuple]]):
    def nlp_fn(text: str) -> MagicMock:
        raw_ents = entities_by_text.get(text, [])
        ents = [_make_ent(t, l, s, e) for t, l, s, e in raw_ents]
        return _make_doc(ents)

    return nlp_fn


class TestReplaceFR:
    def test_single_person(self):
        text = "Marie demande de l'aide."
        nlp = _make_nlp({text: [("Marie", "PER", 0, 5)]})
        with patch("pipeline.anonymize._nlp_for", return_value=nlp):
            mapping, counter = {}, [0]
            result = _replace_persons(text, mapping, counter)
        assert result == "[PER_1] demande de l'aide."
        assert mapping == {"marie": "[PER_1]"}

    def test_two_persons(self):
        text = "Marie demande à Jean de l'aider."
        nlp = _make_nlp({text: [("Marie", "PER", 0, 5), ("Jean", "PER", 16, 20)]})
        with patch("pipeline.anonymize._nlp_for", return_value=nlp):
            mapping, counter = {}, [0]
            result = _replace_persons(text, mapping, counter)
        assert result == "[PER_1] demande à [PER_2] de l'aider."

    def test_same_person_twice(self):
        text = "Jean parle. Jean écoute."
        nlp = _make_nlp({text: [("Jean", "PER", 0, 4), ("Jean", "PER", 12, 16)]})
        with patch("pipeline.anonymize._nlp_for", return_value=nlp):
            mapping, counter = {}, [0]
            result = _replace_persons(text, mapping, counter)
        assert result == "[PER_1] parle. [PER_1] écoute."
        assert counter[0] == 1


class TestReplaceEN:
    def test_single_person(self):
        text = "Alice asks for help."
        nlp = _make_nlp({text: [("Alice", "PER", 0, 5)]})
        with patch("pipeline.anonymize._nlp_for", return_value=nlp):
            mapping, counter = {}, [0]
            result = _replace_persons(text, mapping, counter)
        assert result == "[PER_1] asks for help."

    def test_two_persons(self):
        text = "Alice meets Bob at the crossroads."
        nlp = _make_nlp({text: [("Alice", "PER", 0, 5), ("Bob", "PER", 12, 15)]})
        with patch("pipeline.anonymize._nlp_for", return_value=nlp):
            mapping, counter = {}, [0]
            result = _replace_persons(text, mapping, counter)
        assert result == "[PER_1] meets [PER_2] at the crossroads."


class TestSessionCoherence:
    def test_cross_message_coherence(self):
        msg_user = "Marie demande à Jean de l'aider."
        msg_asst = "Jean acquiesce."
        nlp_user = _make_nlp({msg_user: [("Marie", "PER", 0, 5), ("Jean", "PER", 16, 20)]})
        nlp_asst = _make_nlp({msg_asst: [("Jean", "PER", 0, 4)]})

        def nlp_for_side_effect(text):
            return nlp_user if text == msg_user else nlp_asst

        messages = [
            {"role": "system", "content": "Contexte RP."},
            {"role": "user", "content": msg_user},
            {"role": "assistant", "content": msg_asst},
        ]
        with patch("pipeline.anonymize._nlp_for", side_effect=nlp_for_side_effect):
            result = anonymize_session(messages)

        assert result[0]["content"] == "Contexte RP."
        assert result[1]["content"] == "[PER_1] demande à [PER_2] de l'aider."
        assert result[2]["content"] == "[PER_2] acquiesce."

    def test_system_message_untouched(self):
        messages = [{"role": "system", "content": "Instructions secrètes."}]
        result = anonymize_session(messages)
        assert result[0]["content"] == "Instructions secrètes."

    def test_extra_fields_preserved(self):
        msg = {"role": "user", "content": "Bonjour.", "id": "abc123"}
        nlp = _make_nlp({"Bonjour.": []})
        with patch("pipeline.anonymize._nlp_for", return_value=nlp):
            result = anonymize_session([msg])
        assert result[0]["id"] == "abc123"


class TestNoEntity:
    def test_text_without_persons(self):
        text = "Le soleil se couche sur la forêt."
        nlp = _make_nlp({text: []})
        with patch("pipeline.anonymize._nlp_for", return_value=nlp):
            mapping, counter = {}, [0]
            result = _replace_persons(text, mapping, counter)
        assert result == text
        assert counter[0] == 0

    def test_empty_content(self):
        messages = [{"role": "user", "content": ""}]
        result = anonymize_session(messages)
        assert result[0]["content"] == ""


class TestLangDetect:
    def test_detect_fr(self):
        with patch("pipeline.anonymize.detect", return_value="fr"):
            assert _detect_lang("Bonjour le monde") == "fr"

    def test_detect_en(self):
        with patch("pipeline.anonymize.detect", return_value="en"):
            assert _detect_lang("Hello world") == "en"

    def test_detect_unknown_falls_back_to_fr(self):
        with patch("pipeline.anonymize.detect", return_value="de"):
            assert _detect_lang("Guten Tag") == "fr"

    def test_detect_exception_falls_back_to_fr(self):
        from langdetect import LangDetectException

        with patch("pipeline.anonymize.detect", side_effect=LangDetectException(0, "")):
            assert _detect_lang("???") == "fr"
