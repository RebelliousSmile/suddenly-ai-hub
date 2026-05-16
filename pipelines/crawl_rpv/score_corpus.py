#!/usr/bin/env python3
"""
Scoreur de curation pour corpus RP traduit (sortie de translate_csv_nllb.py).

Calcule trois axes de score par ligne :
  - qualité linguistique (longueur, ponctuation finale, placeholders intacts,
    ratio FR/EN plausible, pas de mots EN résiduels)
  - rééquilibrage par couple (genre, situation) — pondération inverse fréquence
  - déduplication near-duplicate via MinHash + LSH (Jaccard sur 5-grammes
    de mots normalisés)

Sortie : CSV trié par score décroissant, avec colonnes additionnelles
(score, score_quality, weight_balance, is_duplicate, dup_cluster, flags).
Le tri permet de choisir un cutoff manuel (top N, top X%, ou seuil).

Usage:
    python score_corpus.py data/renpy-corpus_en-to-fr_nllb-1.3B.csv
    python score_corpus.py ... --jaccard 0.85 -o data/renpy-corpus_scored.csv
"""
import argparse
import csv
import hashlib
import math
import os
import random
import re
import unicodedata
from collections import Counter, defaultdict
from typing import Iterable

# --- Qualité linguistique ---

MIN_LEN, MAX_LEN = 50, 2000
RATIO_MIN, RATIO_MAX = 0.6, 1.6  # len(FR) / len(EN)

END_PUNCT = re.compile(r'[.!?»"\'\)\]…]\s*$')
PLACEHOLDER = re.compile(r'\[[a-zA-Z_][a-zA-Z0-9_]*\]')
GARBAGE = re.compile(r'(\.{4,}|[\x00-\x08\x0b\x0c\x0e-\x1f])')

# Mots anglais distinctifs (faux positifs minimisés : pas de "et/est/sur/...")
EN_RESIDUAL = re.compile(
    r'\b(?:the|and|with|that|this|what|when|where|why|how|your|you|are|was|'
    r'were|been|being|have|has|had|will|would|could|should|about|from|into|'
    r'their|there|these|those|which|while)\b',
    re.IGNORECASE,
)


def score_quality(content: str, content_fr: str) -> tuple[float, list[str]]:
    """Renvoie (score in [0,1], list of flag strings)."""
    flags = []
    score = 1.0

    L_fr = len(content_fr)
    L_en = len(content) if content else 1
    if L_fr < MIN_LEN:
        score *= 0.0
        flags.append(f"too_short:{L_fr}")
    elif L_fr > MAX_LEN:
        score *= 0.5
        flags.append(f"too_long:{L_fr}")

    ratio = L_fr / L_en
    if ratio < RATIO_MIN:
        score *= 0.4
        flags.append(f"ratio_low:{ratio:.2f}")
    elif ratio > RATIO_MAX:
        score *= 0.6
        flags.append(f"ratio_high:{ratio:.2f}")

    if not END_PUNCT.search(content_fr.rstrip()):
        score *= 0.85
        flags.append("no_end_punct")

    ph_en = set(PLACEHOLDER.findall(content))
    ph_fr = set(PLACEHOLDER.findall(content_fr))
    if ph_en != ph_fr:
        score *= 0.3
        flags.append(f"placeholder_mismatch:en={len(ph_en)}/fr={len(ph_fr)}")

    if GARBAGE.search(content_fr):
        score *= 0.5
        flags.append("garbage_chars")

    en_hits = len(EN_RESIDUAL.findall(content_fr))
    if en_hits >= 3:
        score *= 0.3
        flags.append(f"en_residual:{en_hits}")
    elif en_hits >= 1:
        score *= 0.75
        flags.append(f"en_residual:{en_hits}")

    return score, flags


# --- Balance par (genre, situation) ---

def balance_weights(rows: list[dict]) -> list[float]:
    """Pondération inverse-fréquence douce : log(N_total / N_couple), normalisée."""
    counts = Counter((r.get('genre', ''), r.get('situation', '')) for r in rows)
    N = len(rows)
    weights = []
    for r in rows:
        n = counts[(r.get('genre', ''), r.get('situation', ''))]
        # log naturel borné : valeur 1 pour couple "moyen", >1 pour rare, <1 pour ultra-fréquent
        w = math.log(N / n) / math.log(N / (N / len(counts) or 1))
        weights.append(max(0.3, min(2.0, w)))
    return weights


# --- MinHash + LSH ---

NUM_PERM = 64
LSH_BANDS = 16
LSH_ROWS = NUM_PERM // LSH_BANDS  # 4
SHINGLE_K = 5
MERSENNE_PRIME = (1 << 61) - 1


def normalize_for_shingles(text: str) -> list[str]:
    """Lowercase + strip accents + tokens alphanum."""
    text = unicodedata.normalize('NFKD', text.lower())
    text = ''.join(c for c in text if not unicodedata.combining(c))
    return re.findall(r"[a-z0-9']+", text)


def shingles(tokens: list[str], k: int = SHINGLE_K) -> list[str]:
    if len(tokens) < k:
        return [' '.join(tokens)] if tokens else []
    return [' '.join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)]


def make_perms(num: int, seed: int = 1) -> list[tuple[int, int]]:
    rng = random.Random(seed)
    return [(rng.randint(1, MERSENNE_PRIME - 1), rng.randint(0, MERSENNE_PRIME - 1))
            for _ in range(num)]


def shingle_hash(s: str) -> int:
    return int.from_bytes(hashlib.blake2b(s.encode('utf-8'), digest_size=8).digest(), 'big')


def minhash_signature(shings: list[str], perms: list[tuple[int, int]]) -> tuple[int, ...]:
    if not shings:
        return tuple(MERSENNE_PRIME for _ in perms)
    base = [shingle_hash(s) for s in shings]
    sig = []
    for a, b in perms:
        sig.append(min(((a * h + b) % MERSENNE_PRIME) for h in base))
    return tuple(sig)


def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    return len(a & b) / max(1, len(a | b))


def find_duplicate_clusters(rows: list[dict], quality_scores: list[float],
                             threshold: float = 0.85) -> tuple[list[int], list[bool]]:
    """Renvoie (cluster_id_par_row, is_duplicate_flag).

    Stratégie : MinHash signature -> LSH banding -> candidats -> Jaccard exact ->
    union-find des paires similaires. Représentant = ligne du cluster avec le
    meilleur score qualité (les autres sont marquées is_duplicate=True).
    """
    perms = make_perms(NUM_PERM, seed=1)

    # 1. Pré-calc shingles + signatures
    shing_sets = []
    sigs = []
    for r in rows:
        tokens = normalize_for_shingles(r.get('content_fr', '') or '')
        shs = shingles(tokens)
        shing_sets.append(set(shs))
        sigs.append(minhash_signature(shs, perms))

    # 2. LSH banding -> buckets de candidats
    buckets: dict[tuple, list[int]] = defaultdict(list)
    for idx, sig in enumerate(sigs):
        for b in range(LSH_BANDS):
            key = (b,) + sig[b * LSH_ROWS:(b + 1) * LSH_ROWS]
            buckets[key].append(idx)

    # 3. Pour chaque bucket, vérifier Jaccard exact des candidats
    parent = list(range(len(rows)))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    pairs_checked = set()
    for bucket in buckets.values():
        if len(bucket) < 2:
            continue
        for i in range(len(bucket)):
            for j in range(i + 1, len(bucket)):
                a, b = bucket[i], bucket[j]
                if (a, b) in pairs_checked:
                    continue
                pairs_checked.add((a, b))
                if jaccard(shing_sets[a], shing_sets[b]) >= threshold:
                    union(a, b)

    # 4. Cluster ID + désignation du représentant (meilleur score qualité)
    clusters_members: dict[int, list[int]] = defaultdict(list)
    for idx in range(len(rows)):
        clusters_members[find(idx)].append(idx)

    cluster_id = [0] * len(rows)
    is_duplicate = [False] * len(rows)
    for cid, (root, members) in enumerate(clusters_members.items()):
        if len(members) == 1:
            cluster_id[members[0]] = cid
            continue
        # Représentant = celui avec le meilleur score qualité (tie-break: plus court idx)
        best = max(members, key=lambda i: (quality_scores[i], -i))
        for m in members:
            cluster_id[m] = cid
            if m != best:
                is_duplicate[m] = True

    return cluster_id, is_duplicate


# --- Main ---

def main():
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[1])
    parser.add_argument('input', help='CSV produit par translate_csv_nllb.py')
    parser.add_argument('-o', '--output', default=None,
                        help='CSV de sortie (défaut: dérivé de input)')
    parser.add_argument('--jaccard', type=float, default=0.85,
                        help='Seuil Jaccard pour near-duplicate (défaut: 0.85)')
    parser.add_argument('--keep-skipped', action='store_true',
                        help='Conserver les lignes _skip_reason (par défaut exclues)')
    parser.add_argument('-w', '--weights', default='0.5,0.3,0.2',
                        help='Pondérations quality,balance,non-dup (défaut: 0.5,0.3,0.2)')
    args = parser.parse_args()

    w_q, w_b, w_d = (float(x) for x in args.weights.split(','))

    if args.output is None:
        folder, fname = os.path.split(args.input)
        stem, ext = os.path.splitext(fname)
        out = f"{stem}_scored{ext}"
        args.output = os.path.join(folder, out) if folder else out

    with open(args.input, 'r', encoding='utf-8') as f:
        all_rows = list(csv.DictReader(f))

    if not args.keep_skipped:
        rows = [r for r in all_rows if not r.get('_skip_reason')]
    else:
        rows = all_rows
    print(f"Total lignes input: {len(all_rows)}  "
          f"({len(rows)} retenues après filtre skip)")

    # 1. Qualité
    quality_scores = []
    quality_flags = []
    for r in rows:
        s, fl = score_quality(r.get('content', '') or '', r.get('content_fr', '') or '')
        quality_scores.append(s)
        quality_flags.append('|'.join(fl))
    print(f"Quality median: {sorted(quality_scores)[len(quality_scores) // 2]:.2f}")

    # 2. Balance
    balance = balance_weights(rows)

    # 3. Dédup
    print(f"Calcul MinHash + LSH (Jaccard >= {args.jaccard})...")
    cluster_ids, is_dup = find_duplicate_clusters(rows, quality_scores, args.jaccard)
    n_dup = sum(is_dup)
    n_cluster = len(set(cluster_ids))
    print(f"  Clusters: {n_cluster}  Doublons éliminés: {n_dup}/{len(rows)} "
          f"({100 * n_dup / len(rows):.1f}%)")

    # 4. Score composite
    scored = []
    for r, q, bw, cid, dup, fl in zip(rows, quality_scores, balance,
                                       cluster_ids, is_dup, quality_flags):
        dup_factor = 0.0 if dup else 1.0
        composite = w_q * q + w_b * (bw / 2.0) + w_d * dup_factor
        out_row = dict(r)
        out_row['score'] = round(composite, 4)
        out_row['score_quality'] = round(q, 3)
        out_row['weight_balance'] = round(bw, 3)
        out_row['is_duplicate'] = '1' if dup else '0'
        out_row['dup_cluster'] = cid
        out_row['flags'] = fl
        scored.append(out_row)

    scored.sort(key=lambda r: -r['score'])

    fieldnames = list(scored[0].keys())
    with open(args.output, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(scored)

    # Stats finales
    top10 = scored[:max(1, len(scored) // 10)]
    bot10 = scored[-max(1, len(scored) // 10):]
    print(f"\nDone -> {args.output}")
    print(f"  Lignes utilisables (is_duplicate=0): {len(scored) - n_dup}")
    print(f"  Score top 10%: median {top10[len(top10) // 2]['score']:.3f}")
    print(f"  Score bot 10%: median {bot10[len(bot10) // 2]['score']:.3f}")
    print("\nFlags les plus fréquents :")
    flag_counts = Counter()
    for r in scored:
        for f in r['flags'].split('|'):
            if f:
                # Normalisation : virer la valeur après ':' pour grouper
                key = f.split(':')[0]
                flag_counts[key] += 1
    for flag, n in flag_counts.most_common(10):
        print(f"  {flag:30} {n:>6} ({100 * n / len(scored):.1f}%)")


if __name__ == '__main__':
    main()
