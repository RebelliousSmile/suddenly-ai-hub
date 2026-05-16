#!/usr/bin/env python3
"""
Translation pipeline using NLLB (Facebook's dedicated translation model).

Améliorations vs. version initiale :
- Skip du code Ren'Py (role=system avec directives scene/show/...) -> pass-through
- Skip des lignes déjà en français ou espagnol (heuristique accents/marqueurs)
- Masquage des placeholders [player], [name], etc. avant traduction puis restauration
- Split des contents longs (> 800 chars) en phrases pour éviter la troncature silencieuse
- Modèle NLLB 1.3B (au lieu du 600M distilled) : la RTX 2080 SUPER (8 GB) tient
- 3 passes fusionnées en une seule (les P2/P3 d'origine étaient des no-ops)
"""
import argparse
import csv
import os
import re
import time
import torch
from transformers import pipeline

DEFAULT_INPUT = "data/renpy-corpus-flat.csv"

NLLB_MODEL = "facebook/nllb-200-1.3B"
SRC_LANG = "eng_Latn"
TRG_LANG = "fra_Latn"

_LANG_MAP = {"eng": "en", "fra": "fr", "spa": "es", "deu": "de", "ita": "it"}

def derive_output_path(input_path: str, model_name: str = NLLB_MODEL,
                       src: str = SRC_LANG, tgt: str = TRG_LANG) -> str:
    """data/renpy-corpus-flat.csv -> data/renpy-corpus_en-to-fr_nllb-1.3B.csv

    Retire le suffixe '-flat' du stem s'il est présent, pour ne pas le traîner
    dans tous les fichiers dérivés (le format flat est déjà l'entrée canonique).
    """
    folder, fname = os.path.split(input_path)
    stem, ext = os.path.splitext(fname)
    if stem.endswith("-flat"):
        stem = stem[: -len("-flat")]
    src_short = _LANG_MAP.get(src.split("_")[0], src.split("_")[0])
    tgt_short = _LANG_MAP.get(tgt.split("_")[0], tgt.split("_")[0])
    model_id = model_name.rsplit("/", 1)[-1].replace("nllb-200-", "nllb-")
    out_name = f"{stem}_{src_short}-to-{tgt_short}_{model_id}{ext}"
    return f"{folder}/{out_name}" if folder else out_name
BATCH_SIZE = 8
MAX_LEN = 1024
SPLIT_THRESHOLD = 800

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Détection contenu non traduisible ---

# Tokens Ren'Py typiques : scene/show/hide/play/stop/with/jump/call/menu/label, ou ligne $...
RENPY_KEYWORDS = re.compile(
    r'\b(?:scene|show|hide|play|stop|with|jump|call|menu|label|init|window|return|pause)\b',
    re.IGNORECASE,
)

def is_renpy_code(content: str) -> bool:
    """True si le content ressemble à du markup Ren'Py plutôt qu'à du dialogue."""
    if not content:
        return False
    hits = len(RENPY_KEYWORDS.findall(content))
    # Heuristique : au moins 2 keywords ET ratio keywords/mots > 0.15
    words = max(1, len(content.split()))
    return hits >= 2 and (hits / words) > 0.15

# Accents/diacritiques typiques FR (et ES en bonus)
FR_MARKERS = set("éèêëàâäçîïôöûüùÿœæÉÈÊËÀÂÄÇÎÏÔÖÛÜÙŸŒÆ")
ES_MARKERS = set("ñÑ¿¡")
# Mots fonction discriminants FR (utile quand la ligne n'a pas d'accent).
# Retirés volontairement : mon/ma/ses/ton/ta/son/sa (collisions avec mots EN courants
# type "C'mon", "Macy", "season").
FR_STOPWORDS = re.compile(
    r'\b(?:le|la|les|des|une?|du|aux|que|qui|est|sont|pour|avec|dans|sur|mais|alors|donc|cette?|ces|nous|vous|elles?)\b',
    re.IGNORECASE,
)

# Mots distinctifs FR (très peu probables en anglais)
FR_DISTINCTIVE = re.compile(
    r'\b(?:bonjour|bonsoir|merci|oui|aujourd|pourquoi|quelqu|toujours|jamais|vraiment|comment|déjà|peut-?être|tu\s+(?:es|as|vas|veux|peux|fais|sais|dois))\b',
    re.IGNORECASE,
)

# Marqueurs FR forts : si présents, on classe FR sans ambiguïté
FR_STRONG_MARKERS = re.compile(
    r'^(?:Contexte\s*:|Genre\s*:|Situation\s*:)',
    re.IGNORECASE,
)

def looks_already_translated(content: str) -> bool:
    """True si le content est probablement déjà en FR ou ES (skip translation)."""
    if not content or len(content) < 3:
        return False
    if FR_STRONG_MARKERS.search(content):
        return True
    if any(c in ES_MARKERS for c in content):
        return True
    if any(c in FR_MARKERS for c in content):
        return True
    if FR_DISTINCTIVE.search(content):
        return True
    # Pas d'accent : stopwords FR. Seuil prudent : >=3 matches ET densité > 20%
    matches = FR_STOPWORDS.findall(content)
    words = max(1, len(content.split()))
    return len(matches) >= 3 and (len(matches) / words) > 0.20

# --- Masquage des placeholders ---

PLACEHOLDER_RE = re.compile(r'\[[a-zA-Z_][a-zA-Z0-9_]*\]')

def mask_placeholders(text: str):
    """Remplace [player] -> __PH0__, etc. Retourne (masked, mapping)."""
    mapping = {}
    counter = [0]
    def repl(m):
        token = f"__PH{counter[0]}__"
        mapping[token] = m.group(0)
        counter[0] += 1
        return token
    return PLACEHOLDER_RE.sub(repl, text), mapping

def unmask_placeholders(text: str, mapping: dict) -> str:
    for token, original in mapping.items():
        text = text.replace(token, original)
    return text

# --- Split phrases pour les longs contents ---

SENTENCE_END = re.compile(r'(?<=[.!?])\s+')

def split_long(text: str, max_chunk: int = SPLIT_THRESHOLD):
    """Split par phrases si trop long. Retourne liste de chunks."""
    if len(text) <= max_chunk:
        return [text]
    sentences = SENTENCE_END.split(text)
    chunks, current = [], ""
    for s in sentences:
        if len(current) + len(s) + 1 <= max_chunk:
            current = (current + " " + s).strip() if current else s
        else:
            if current:
                chunks.append(current)
            # Si une seule phrase fait > max_chunk : on l'accepte telle quelle
            # (NLLB tronquera avec max_length, mais c'est explicite ici)
            current = s
    if current:
        chunks.append(current)
    return chunks

# --- Pipeline ---

def main():
    parser = argparse.ArgumentParser(description="NLLB translation pipeline pour corpus Ren'Py")
    parser.add_argument("input", nargs="?", default=DEFAULT_INPUT,
                        help=f"CSV d'entrée (défaut: {DEFAULT_INPUT})")
    parser.add_argument("-o", "--output", default=None,
                        help="CSV de sortie (défaut: dérivé de input)")
    args = parser.parse_args()

    input_file = args.input
    output_file = args.output or derive_output_path(input_file)
    if not os.path.exists(input_file):
        raise SystemExit(f"Input introuvable: {input_file}")

    print(f"Input:  {input_file}")
    print(f"Output: {output_file}")
    print(f"Loading NLLB model: {NLLB_MODEL}")
    print(f"Device: {DEVICE}")
    t0 = time.time()
    translator = pipeline(
        "translation",
        model=NLLB_MODEL,
        src_lang=SRC_LANG,
        tgt_lang=TRG_LANG,
        device=0 if DEVICE == "cuda" else -1,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s")

    with open(input_file, "r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(f"Total entries: {len(rows)}")

    # Classification
    to_translate = []  # liste de (row_idx, chunk_idx, n_chunks, masked_text, mapping)
    skipped_renpy = 0
    skipped_already = 0
    for i, row in enumerate(rows):
        content = row.get("content", "") or ""
        if row.get("role") == "system" and is_renpy_code(content):
            row["content_fr"] = content
            row["_skip_reason"] = "renpy_code"
            skipped_renpy += 1
            continue
        if looks_already_translated(content):
            row["content_fr"] = content
            row["_skip_reason"] = "already_fr_es"
            skipped_already += 1
            continue
        # Sinon : à traduire
        masked, mapping = mask_placeholders(content)
        chunks = split_long(masked)
        for ci, chunk in enumerate(chunks):
            to_translate.append((i, ci, len(chunks), chunk, mapping))
        row["content_fr"] = None  # rempli plus tard
        row["_skip_reason"] = ""

    print(f"To translate: {len(to_translate)} chunks across {sum(1 for r in rows if r.get('_skip_reason') == '')} rows")
    print(f"Skipped (Ren'Py code): {skipped_renpy}")
    print(f"Skipped (already FR/ES): {skipped_already}")

    # Traduction par batch
    translations_by_row = {}  # row_idx -> dict(chunk_idx -> translated_text)
    t_start = time.time()
    total = len(to_translate)

    for i in range(0, total, BATCH_SIZE):
        batch = to_translate[i:i + BATCH_SIZE]
        texts = [item[3] for item in batch]
        try:
            results = translator(texts, batch_size=BATCH_SIZE, max_length=MAX_LEN)
            translated = [r["translation_text"] for r in results]
        except Exception as e:
            print(f"  Batch error at {i}: {e}")
            translated = texts  # fallback : garder l'original
        for (row_idx, chunk_idx, _n, _txt, _map), out in zip(batch, translated):
            translations_by_row.setdefault(row_idx, {})[chunk_idx] = out

        if (i // BATCH_SIZE) % 50 == 0 and i > 0:
            elapsed = time.time() - t_start
            rate = i / elapsed
            eta = (total - i) / rate if rate > 0 else 0
            print(f"  {i}/{total} chunks ({rate:.1f}/s, ETA {eta/60:.1f}min)")

    # Reconstruction : recoller les chunks + restaurer placeholders + cleanup
    code_fence_re = re.compile(r'^```.*?\n(.*?)\n```$', re.DOTALL)
    rows_translated_count = 0
    for i, row in enumerate(rows):
        if row.get("_skip_reason"):
            continue
        chunks_dict = translations_by_row.get(i, {})
        if not chunks_dict:
            row["content_fr"] = row["content"]  # fallback
            continue
        ordered = [chunks_dict[ci] for ci in sorted(chunks_dict.keys())]
        merged = " ".join(ordered).strip()
        # Récupérer le mapping (tous les chunks d'une row partagent le même mapping)
        # On retrouve via to_translate
        merged = code_fence_re.sub(r'\1', merged)
        # Trouver le mapping pour cette row
        for (row_idx, _ci, _n, _txt, mapping) in to_translate:
            if row_idx == i:
                merged = unmask_placeholders(merged, mapping)
                break
        row["content_fr"] = merged
        rows_translated_count += 1

    # Écriture finale
    fieldnames = ["genre", "situation", "role", "content", "content_fr"]
    with open(output_file, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    elapsed = time.time() - t_start
    print(f"\nDone in {elapsed/60:.1f}min")
    print(f"  Translated: {rows_translated_count} rows")
    print(f"  Skipped Ren'Py: {skipped_renpy}")
    print(f"  Skipped FR/ES: {skipped_already}")
    print(f"Output: {output_file}")

if __name__ == "__main__":
    main()
