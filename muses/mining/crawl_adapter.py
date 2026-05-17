"""Adapter qui lit les sorties JSONL du pipeline crawl_rpv (et formats RP
similaires) et produit des row dicts prêts pour l'ingestion.

T12 du technical-plan.

Le format source attendu : JSONL où chaque ligne est un objet `{"messages":
[{"role": ..., "content": ...}, ...], "metadata": {...}}` — c'est ce que
produisent les corpus Ren'Py et test-dataset-rp.jsonl déjà présents.

L'extraction par défaut produit des rows de niveau `fragment` à partir des
messages `assistant` (les réponses RP générées). Les messages `system` et
`user` ne sont pas extraits — ce sont du contexte, pas du contenu narratif
publiable comme row.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterator

from muses.mining.anonymization import anonymize_text


# Découpage en phrases : approximation suffisante pour le bootstrap, ne
# tente pas de gérer les cas pointus (abréviations, etc.). À muscler en
# v2 avec un tokenizer adapté.
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?…»])\s+(?=[A-ZÀ-Ý«—\"\*])")


def parse_rp_dataset_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    """Itère les entrées d'un JSONL RP au format `{messages, metadata}`."""
    path = Path(path)
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _split_into_fragments(text: str, *, min_chars: int = 30) -> list[str]:
    """Découpe un texte en fragments (phrases). Filtre les trop courts."""
    candidates = _SENTENCE_SPLIT.split(text)
    return [c.strip() for c in candidates if len(c.strip()) >= min_chars]


def extract_fragments_from_rp_dataset(
    jsonl_path: Path,
    *,
    tags: dict[str, list[str]],
    source: str = "bootstrap",
    instance_id: str | None = None,
    user_id: str | None = None,
    anonymize: bool = True,
    min_chars: int = 30,
) -> list[dict[str, Any]]:
    """Lit un JSONL RP et produit la liste des row dicts (niveau `fragment`).

    Tous les fragments produits partagent les mêmes `tags` — le caller est
    censé filtrer le JSONL en amont selon les axes pertinents (ou tagger par
    autre voie).

    `source` doit être `bootstrap` ou `mined` pour ne pas exiger de
    signature ActivityPub côté ingestion.
    """
    rows: list[dict[str, Any]] = []
    for entry in parse_rp_dataset_jsonl(jsonl_path):
        messages = entry.get("messages", [])
        for msg in messages:
            if msg.get("role") != "assistant":
                continue
            content = msg.get("content", "")
            for fragment in _split_into_fragments(content, min_chars=min_chars):
                final_text = (
                    anonymize_text(fragment).text if anonymize else fragment
                )
                rows.append({
                    "level": "fragment",
                    "tags": tags,
                    "content": {
                        "text": final_text,
                        "char_pov": "neutral",
                    },
                    "source": source,
                    "user_id": user_id,
                    "instance_id": instance_id,
                    "signature": None,
                })
    return rows
