import json
import textwrap
from pathlib import Path

import pytest

from pipeline.format_corpus import (
    SYSTEM_PROMPT_RP,
    _is_valid_session,
    _parse_dialogue,
    _parse_jsonl,
    _parse_narrative,
    convert,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _long(text: str, repeat: int = 80) -> str:
    """Répète le texte pour dépasser le seuil MIN_WORDS (150 mots)."""
    return " ".join([text] * repeat)


# ---------------------------------------------------------------------------
# _is_valid_session
# ---------------------------------------------------------------------------

class TestIsValidSession:
    def test_valid_two_turns(self):
        msgs = [
            {"role": "user", "content": _long("action du joueur")},
            {"role": "assistant", "content": _long("réponse du MJ")},
        ]
        assert _is_valid_session(msgs) is True

    def test_valid_with_system(self):
        msgs = [
            {"role": "system", "content": "Contexte RP."},
            {"role": "user", "content": _long("action du joueur")},
            {"role": "assistant", "content": _long("réponse du MJ")},
        ]
        assert _is_valid_session(msgs) is True

    def test_empty(self):
        assert _is_valid_session([]) is False

    def test_only_system(self):
        assert _is_valid_session([{"role": "system", "content": "x"}]) is False

    def test_bad_alternation_user_user(self):
        msgs = [
            {"role": "user", "content": _long("un")},
            {"role": "user", "content": _long("deux")},
        ]
        assert _is_valid_session(msgs) is False

    def test_bad_alternation_starts_assistant(self):
        msgs = [
            {"role": "assistant", "content": _long("réponse")},
            {"role": "user", "content": _long("action")},
        ]
        assert _is_valid_session(msgs) is False

    def test_too_short(self):
        msgs = [
            {"role": "user", "content": "court"},
            {"role": "assistant", "content": "réponse"},
        ]
        assert _is_valid_session(msgs) is False


# ---------------------------------------------------------------------------
# _parse_dialogue
# ---------------------------------------------------------------------------

class TestParseDialogue:
    def _build(self, *lines: str) -> str:
        return "\n".join(lines)

    def test_two_speakers_simple(self):
        text = self._build(
            f"Alice: {_long('action')}",
            f"Bob: {_long('réponse')}",
        )
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 1
        assert sessions[0][0]["role"] == "user"
        assert sessions[0][1]["role"] == "assistant"

    def test_markdown_bold_speaker(self):
        text = self._build(
            f"**Alice** : {_long('action')}",
            f"**Bob** : {_long('réponse')}",
        )
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 1

    def test_bracket_speaker(self):
        text = self._build(
            f"[Alice] : {_long('action')}",
            f"[Bob] : {_long('réponse')}",
        )
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 1

    def test_multiple_sessions_separated_by_blank_line(self):
        session1 = f"Alice: {_long('action un')}\nBob: {_long('réponse un')}"
        session2 = f"Alice: {_long('action deux')}\nBob: {_long('réponse deux')}"
        text = session1 + "\n\n" + session2
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 2

    def test_consecutive_same_role_merged(self):
        text = self._build(
            f"Alice: {_long('premier')}",
            f"Alice: {_long('deuxième')}",
            f"Bob: {_long('réponse')}",
        )
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 1
        assert sessions[0][0]["role"] == "user"
        # Les deux lignes Alice sont fusionnées en un seul message
        assert "premier" in sessions[0][0]["content"]
        assert "deuxième" in sessions[0][0]["content"]

    def test_too_short_ignored(self):
        text = "Alice: action\nBob: réponse"
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 0

    def test_single_speaker_ignored(self):
        text = f"Alice: {_long('action seule')}"
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 0

    def test_third_speaker_mapped_to_alternating_role(self):
        text = self._build(
            f"Alice: {_long('action alice')}",
            f"Bob: {_long('réponse bob')}",
            f"Charlie: {_long('action charlie')}",
            f"Bob: {_long('réponse bob encore')}",
        )
        sessions = list(_parse_dialogue(text))
        assert len(sessions) == 1
        # Alice → user, Bob → assistant, Charlie → user (3e locuteur, alternance)
        assert sessions[0][0]["role"] == "user"
        assert sessions[0][1]["role"] == "assistant"
        assert sessions[0][2]["role"] == "user"


# ---------------------------------------------------------------------------
# _parse_narrative
# ---------------------------------------------------------------------------

class TestParseNarrative:
    def test_basic_four_paragraphs(self):
        paras = [_long(f"paragraphe {i}") for i in range(4)]
        text = "\n\n".join(paras)
        sessions = list(_parse_narrative(text, window=4))
        assert len(sessions) == 1
        assert sessions[0][0]["role"] == "user"
        assert sessions[0][1]["role"] == "assistant"

    def test_single_paragraph_ignored(self):
        text = _long("un seul paragraphe")
        sessions = list(_parse_narrative(text))
        assert len(sessions) == 0

    def test_window_splitting(self):
        paras = [_long(f"paragraphe {i}") for i in range(12)]
        text = "\n\n".join(paras)
        sessions = list(_parse_narrative(text, window=4))
        assert len(sessions) == 3

    def test_consecutive_same_role_merged(self):
        # Avec window=3, on a : user, assistant, user → user consécutifs sur le bord
        paras = [_long(f"para {i}") for i in range(3)]
        text = "\n\n".join(paras)
        sessions = list(_parse_narrative(text, window=6))
        # 3 paras : user(0), assistant(1), user(2) → pas de doubles consécutifs
        assert sessions[0][2]["role"] == "user"

    def test_non_divisible_paragraph_count(self):
        # 13 paragraphes, window=4 → 3 sessions complètes (12 paras) + 1 chunk terminal (1 para)
        # Le chunk terminal de 1 para doit être ignoré sans planter
        paras = [_long(f"paragraphe {i}") for i in range(13)]
        text = "\n\n".join(paras)
        sessions = list(_parse_narrative(text, window=4))
        assert len(sessions) == 3


# ---------------------------------------------------------------------------
# _parse_jsonl
# ---------------------------------------------------------------------------

class TestParseJsonl:
    def _line(self, messages: list) -> str:
        return json.dumps({"messages": messages}, ensure_ascii=False)

    def test_valid_session(self):
        line = self._line([
            {"role": "user", "content": _long("action")},
            {"role": "assistant", "content": _long("réponse")},
        ])
        sessions = list(_parse_jsonl(line))
        assert len(sessions) == 1

    def test_invalid_json_skipped(self):
        sessions = list(_parse_jsonl("{broken json}"))
        assert len(sessions) == 0

    def test_missing_messages_key_skipped(self):
        sessions = list(_parse_jsonl('{"text": "foo"}'))
        assert len(sessions) == 0

    def test_invalid_structure_skipped(self):
        line = self._line([
            {"role": "assistant", "content": _long("commence par assistant")},
            {"role": "user", "content": _long("puis user")},
        ])
        sessions = list(_parse_jsonl(line))
        assert len(sessions) == 0

    def test_multiple_lines(self):
        valid = self._line([
            {"role": "user", "content": _long("action")},
            {"role": "assistant", "content": _long("réponse")},
        ])
        lines = valid + "\n" + "{bad}" + "\n" + valid
        sessions = list(_parse_jsonl(lines))
        assert len(sessions) == 2


# ---------------------------------------------------------------------------
# convert() — intégration
# ---------------------------------------------------------------------------

class TestConvert:
    def test_dialogue_with_system(self, tmp_path: Path):
        src = tmp_path / "corpus.txt"
        src.write_text(
            f"Alice: {_long('action')}\nBob: {_long('réponse')}\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.jsonl"
        count = convert(src, out, "dialogue", SYSTEM_PROMPT_RP)
        assert count == 1
        obj = json.loads(out.read_text(encoding="utf-8").strip())
        assert obj["messages"][0]["role"] == "system"

    def test_dialogue_no_system(self, tmp_path: Path):
        src = tmp_path / "corpus.txt"
        src.write_text(
            f"Alice: {_long('action')}\nBob: {_long('réponse')}\n",
            encoding="utf-8",
        )
        out = tmp_path / "out.jsonl"
        count = convert(src, out, "dialogue", system_prompt=None)
        assert count == 1
        obj = json.loads(out.read_text(encoding="utf-8").strip())
        assert obj["messages"][0]["role"] == "user"

    def test_empty_directory(self, tmp_path: Path):
        out = tmp_path / "out.jsonl"
        count = convert(tmp_path, out, "dialogue", system_prompt=None)
        assert count == 0

    def test_glob_filter(self, tmp_path: Path):
        (tmp_path / "corpus.txt").write_text(
            f"Alice: {_long('action')}\nBob: {_long('réponse')}\n", encoding="utf-8"
        )
        (tmp_path / "notes.md").write_text("# notes\n", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        count = convert(tmp_path, out, "dialogue", system_prompt=None, glob_pattern="*.txt")
        assert count == 1

    def test_output_dir_created(self, tmp_path: Path):
        src = tmp_path / "corpus.txt"
        src.write_text(
            f"Alice: {_long('action')}\nBob: {_long('réponse')}\n", encoding="utf-8"
        )
        out = tmp_path / "subdir" / "out.jsonl"
        convert(src, out, "dialogue", system_prompt=None)
        assert out.exists()

    def test_empty_file_produces_zero_sessions(self, tmp_path: Path):
        src = tmp_path / "empty.txt"
        src.write_text("", encoding="utf-8")
        out = tmp_path / "out.jsonl"
        count = convert(src, out, "dialogue", system_prompt=None)
        assert count == 0
