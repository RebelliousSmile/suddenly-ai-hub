from __future__ import annotations

from typing import Optional

import spacy
from langdetect import detect, LangDetectException
from langdetect.detector_factory import DetectorFactory

DetectorFactory.seed = 0

_nlp_fr: Optional[spacy.language.Language] = None
_nlp_en: Optional[spacy.language.Language] = None


def _get_nlp_fr() -> spacy.language.Language:
    global _nlp_fr
    if _nlp_fr is None:
        _nlp_fr = spacy.load("fr_core_news_md")
    return _nlp_fr


def _get_nlp_en() -> spacy.language.Language:
    global _nlp_en
    if _nlp_en is None:
        _nlp_en = spacy.load("en_core_web_md")
    return _nlp_en


def _detect_lang(text: str) -> str:
    try:
        lang = detect(text)
        return lang if lang in ("fr", "en") else "fr"
    except LangDetectException:
        return "fr"


def _nlp_for(text: str) -> spacy.language.Language:
    return _get_nlp_en() if _detect_lang(text) == "en" else _get_nlp_fr()


def _replace_persons(text: str, mapping: dict[str, str], counter: list[int]) -> str:
    if not text or not text.strip():
        return text

    nlp = _nlp_for(text)
    doc = nlp(text)

    persons = [ent for ent in doc.ents if ent.label_ == "PER"]
    if not persons:
        return text

    result = []
    cursor = 0
    for ent in persons:
        result.append(text[cursor : ent.start_char])
        key = ent.text.lower()
        if key not in mapping:
            counter[0] += 1
            mapping[key] = f"[PER_{counter[0]}]"
        result.append(mapping[key])
        cursor = ent.end_char
    result.append(text[cursor:])
    return "".join(result)


def anonymize_session(messages: list[dict]) -> list[dict]:
    mapping: dict[str, str] = {}
    counter: list[int] = [0]
    result = []
    for msg in messages:
        role = msg.get("role", "")
        content = msg.get("content", "")
        if role == "system" or not isinstance(content, str):
            result.append(dict(msg))
            continue
        anonymized = _replace_persons(content, mapping, counter)
        result.append({**msg, "content": anonymized})
    return result
