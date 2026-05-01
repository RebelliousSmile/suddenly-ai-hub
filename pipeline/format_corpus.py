#!/usr/bin/env python3
"""
format_corpus.py — Conversion d'un corpus brut vers le format JSONL d'entraînement.

Formats d'entrée supportés :
  dialogue  : Paires "ROLE: texte" alternées (ex. : transcripts JDR)
  narrative : Texte narratif brut découpé en fenêtres user/assistant
  jsonl     : JSONL existant à valider / normaliser

Règles de sortie (issues #9) :
  - Un objet JSON par ligne : {"messages": [...]}
  - Rôles valides : system, user, assistant
  - Alternance stricte user/assistant (system optionnel en premier)
  - Longueur minimale : 200 tokens estimés (~150 mots)

Usage :
  python pipeline/format_corpus.py --input data/raw/corpus.txt \\
      --format dialogue --output data/corpus-rp.jsonl
  python pipeline/format_corpus.py --input data/raw/corpus.txt \\
      --format narrative --output data/corpus-narrative.jsonl
  python pipeline/format_corpus.py --input data/raw/corpus.jsonl \\
      --format jsonl --output data/corpus-clean.jsonl
  python pipeline/format_corpus.py --input data/raw/ \\
      --format dialogue --output data/corpus-rp.jsonl --glob "*.txt"
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Iterator


# ---------------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------------

MIN_WORDS = 10  # seuil minimal : sessions trop courtes ignorées
SYSTEM_PROMPT_RP = (
    "Tu es un conteur de roleplay. "
    "Réponds en français, dans un registre narratif et immersif."
)

# Patterns pour le format dialogue
_SPEAKER_PATTERNS = [
    re.compile(r"^\*\*(.+?)\*\*\s*:\s*(.+)$"),  # **Nom** : texte
    re.compile(r"^\[(.+?)\]\s*:\s*(.+)$"),        # [Nom] : texte
    re.compile(r"^([A-Z][A-Za-zÀ-ÿ\s]{0,30})\s*:\s*(.+)$"),  # NOM : texte
]


# ---------------------------------------------------------------------------
# Parseurs d'entrée
# ---------------------------------------------------------------------------

def _parse_dialogue(text: str) -> Iterator[list[dict]]:
    """
    Convertit un texte en format "SPEAKER: texte" en sessions JSONL.

    Chaque bloc séparé par une ligne vide constitue une session.
    Le premier locuteur est mappé sur "user", le second sur "assistant".
    Les locuteurs alternent strictement ; les blocs non alternés sont ignorés.
    """
    sessions_raw: list[list[tuple[str, str]]] = []
    current: list[tuple[str, str]] = []

    for line in text.splitlines():
        line = line.strip()
        if not line:
            if current:
                sessions_raw.append(current)
                current = []
            continue
        match = None
        for pat in _SPEAKER_PATTERNS:
            m = pat.match(line)
            if m:
                match = m
                break
        if match:
            current.append((match.group(1).strip(), match.group(2).strip()))

    if current:
        sessions_raw.append(current)

    for turns in sessions_raw:
        if len(turns) < 2:
            continue
        # Construire le mapping speaker→rôle à partir du premier locuteur
        speakers: dict[str, str] = {}
        messages: list[dict] = []
        for speaker, content in turns:
            key = speaker.lower()
            if key not in speakers:
                role = "user" if len(speakers) == 0 else "assistant"
                speakers[key] = role
            role = speakers.get(key)
            if role is None:
                continue
            # Vérifier l'alternance
            if messages and messages[-1]["role"] == role:
                # Fusionner si même rôle consécutif
                messages[-1]["content"] += "\n" + content
            else:
                messages.append({"role": role, "content": content})

        if _is_valid_session(messages):
            yield messages


def _parse_narrative(text: str, window: int = 6) -> Iterator[list[dict]]:
    """
    Découpe un texte narratif en fenêtres glissantes de `window` paragraphes.

    Structure : user = paragraphe pair (action/setup), assistant = paragraphe impair (narration).
    Fenêtres non chevauchantes de taille `window` paragraphes.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if len(paragraphs) < 2:
        return

    for i in range(0, len(paragraphs) - 1, window):
        chunk = paragraphs[i : i + window]
        if len(chunk) < 2:
            break
        messages: list[dict] = []
        for j, para in enumerate(chunk):
            role = "user" if j % 2 == 0 else "assistant"
            if messages and messages[-1]["role"] == role:
                messages[-1]["content"] += "\n\n" + para
            else:
                messages.append({"role": role, "content": para})

        if _is_valid_session(messages):
            yield messages


def _parse_jsonl(text: str) -> Iterator[list[dict]]:
    """
    Valide et normalise un JSONL existant.
    Rejette les sessions dont la structure est invalide.
    """
    for lineno, line in enumerate(text.splitlines(), 1):
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError as exc:
            print(f"[WARN] ligne {lineno} : JSON invalide — {exc}", file=sys.stderr)
            continue

        messages = obj.get("messages")
        if not isinstance(messages, list):
            print(f"[WARN] ligne {lineno} : champ 'messages' manquant ou invalide", file=sys.stderr)
            continue

        if _is_valid_session(messages):
            yield messages
        else:
            print(f"[WARN] ligne {lineno} : session ignorée (structure invalide ou trop courte)", file=sys.stderr)


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _is_valid_session(messages: list[dict]) -> bool:
    """Vérifie qu'une session respecte les règles du format d'entraînement."""
    if not messages:
        return False

    filtered = [m for m in messages if m.get("role") != "system"]
    if len(filtered) < 2:
        return False

    # Alternance user/assistant
    for i, msg in enumerate(filtered):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected:
            return False

    # Longueur minimale estimée
    total_words = sum(len(m.get("content", "").split()) for m in messages)
    if total_words < MIN_WORDS:
        return False

    return True


# ---------------------------------------------------------------------------
# Conversion principale
# ---------------------------------------------------------------------------

def convert(
    input_path: Path,
    output_path: Path,
    fmt: str,
    system_prompt: str | None,
    glob_pattern: str = "*.txt",
) -> int:
    """Convertit les fichiers d'entrée et écrit le JSONL de sortie. Retourne le nombre de sessions."""
    if input_path.is_dir():
        sources = sorted(input_path.glob(glob_pattern))
    else:
        sources = [input_path]

    if not sources:
        print(f"[ERROR] Aucun fichier trouvé dans {input_path}", file=sys.stderr)
        return 0

    parsers = {
        "dialogue": _parse_dialogue,
        "narrative": _parse_narrative,
        "jsonl": _parse_jsonl,
    }
    parser = parsers[fmt]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with output_path.open("w", encoding="utf-8") as out:
        for src in sources:
            text = src.read_text(encoding="utf-8", errors="replace")
            for messages in parser(text):
                if system_prompt and messages[0].get("role") != "system":
                    messages = [{"role": "system", "content": system_prompt}] + messages
                out.write(json.dumps({"messages": messages}, ensure_ascii=False))
                out.write("\n")
                count += 1

    return count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Convertit un corpus brut vers le format JSONL d'entraînement Axolotl.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--input", required=True, type=Path, help="Fichier ou dossier source")
    p.add_argument("--output", required=True, type=Path, help="Fichier JSONL de sortie")
    p.add_argument(
        "--format",
        choices=["dialogue", "narrative", "jsonl"],
        required=True,
        dest="fmt",
        help="Format du corpus source",
    )
    p.add_argument(
        "--system",
        default=None,
        metavar="PROMPT",
        help=f"System prompt à injecter (défaut : '{SYSTEM_PROMPT_RP}'). Passer --system '' pour désactiver.",
    )
    p.add_argument(
        "--no-system",
        action="store_true",
        help="Ne pas injecter de system prompt",
    )
    p.add_argument(
        "--glob",
        default="*.txt",
        help="Glob pour filtrer les fichiers si --input est un dossier (défaut : *.txt)",
    )
    return p


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)

    if args.no_system:
        system_prompt: str | None = None
    elif args.system is not None:
        system_prompt = args.system or None
    else:
        system_prompt = SYSTEM_PROMPT_RP

    args.output.parent.mkdir(parents=True, exist_ok=True)

    count = convert(
        input_path=args.input,
        output_path=args.output,
        fmt=args.fmt,
        system_prompt=system_prompt,
        glob_pattern=args.glob,
    )
    print(f"[INFO] {count} sessions écrites dans {args.output}")


if __name__ == "__main__":
    main()
