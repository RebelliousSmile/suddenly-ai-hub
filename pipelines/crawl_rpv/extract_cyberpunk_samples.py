#!/usr/bin/env python3
"""
Extrait des passages variés de 3 oeuvres cyberpunk anglaises (CC) pour
benchmark de traduction NLLB vs LLM. Lit les sources brutes téléchargées
dans ~/lora-fr/bench/cyberpunk/sources/ et écrit un CSV prêt pour le
benchmark.

Usage:
    python extract_cyberpunk_samples.py [-n 5] [-s 42]
"""
import argparse
import csv
import os
import random
import re
from pathlib import Path

from bs4 import BeautifulSoup

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "data" / "bench" / "cyberpunk"
SRC_DIR = BENCH_DIR / "sources"
OUT_CSV = BENCH_DIR / "extracted" / "bench_cyberpunk_en.csv"

MIN_CHARS = 600
MAX_CHARS = 1800

# Headers et patterns récurrents à éliminer
WATTS_NOISE = re.compile(r'^(Peter Watts|Blindsight)\s+\d+', re.MULTILINE)
PAGE_NUMBER = re.compile(r'^\s*\d+\s*$', re.MULTILINE)
MULTI_SPACE = re.compile(r'[ \t]+')


def normalize(text: str) -> str:
    text = MULTI_SPACE.sub(' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()


def paragraphs_from_text(text: str) -> list[str]:
    """Reflow : remerge lignes coupées, splite sur lignes vides."""
    text = WATTS_NOISE.sub('', text)
    text = PAGE_NUMBER.sub('', text)
    # Une ligne vide = nouveau paragraphe ; sinon on rejoint
    blocks = [b.strip() for b in re.split(r'\n\s*\n', text) if b.strip()]
    # Dans chaque bloc, remettre les retours-ligne en espaces (reflow)
    return [normalize(b.replace('\n', ' ')) for b in blocks]


def load_watts() -> list[str]:
    text = (SRC_DIR / "blindsight.txt").read_text(encoding='utf-8', errors='ignore')
    return paragraphs_from_text(text)


def load_doctorow() -> list[str]:
    text = (SRC_DIR / "down-and-out.txt").read_text(encoding='utf-8', errors='ignore')
    # Le fichier a souvent un long header de licence à virer
    # Heuristique : chercher la fin du préambule (souvent "Chapter 1" ou similaire)
    marker = re.search(r'\n\s*(Chapter\s+1|Part\s+I|PROLOGUE)\b', text, re.IGNORECASE)
    if marker:
        text = text[marker.start():]
    return paragraphs_from_text(text)


def load_stross() -> list[str]:
    html = (SRC_DIR / "accelerando.html").read_text(encoding='utf-8', errors='ignore')
    soup = BeautifulSoup(html, 'lxml')
    for tag in soup(['script', 'style', 'head', 'nav']):
        tag.decompose()
    # Récupérer tous les <p> du body
    ps = [p.get_text(' ', strip=True) for p in soup.find_all('p')]
    return [normalize(p) for p in ps if p.strip()]


def pick(paragraphs: list[str], n: int, seed: int) -> list[str]:
    eligible = [p for p in paragraphs if MIN_CHARS <= len(p) <= MAX_CHARS]
    rng = random.Random(seed)
    rng.shuffle(eligible)
    return eligible[:n]


def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('-n', '--samples', type=int, default=5,
                        help='Nombre de passages par auteur (défaut: 5)')
    parser.add_argument('-s', '--seed', type=int, default=42,
                        help='Seed RNG pour reproductibilité (défaut: 42)')
    parser.add_argument('-o', '--output', default=str(OUT_CSV),
                        help=f'CSV de sortie (défaut: {OUT_CSV})')
    args = parser.parse_args()

    loaders = [
        ('watts', 'Blindsight', load_watts),
        ('doctorow', 'Down and Out in the Magic Kingdom', load_doctorow),
        ('stross', 'Accelerando', load_stross),
    ]

    rows = []
    for author, work, loader in loaders:
        paras = loader()
        eligible = [p for p in paras if MIN_CHARS <= len(p) <= MAX_CHARS]
        sample = pick(paras, args.samples, args.seed)
        print(f"{author:10} : {len(paras):5} paragraphes, "
              f"{len(eligible):4} éligibles, {len(sample)} retenus")
        for p in sample:
            rows.append({
                'author': author,
                'work': work,
                'role': 'narration',
                'content': p,  # nom 'content' pour réutiliser translate_csv_nllb.py
            })

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['author', 'work', 'role', 'content'])
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nDone. {len(rows)} passages -> {out}")


if __name__ == '__main__':
    main()
