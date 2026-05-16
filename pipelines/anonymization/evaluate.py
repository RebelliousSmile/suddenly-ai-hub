#!/usr/bin/env python3
"""
evaluate.py — Compare les sorties d'un modèle fine-tuné contre le dataset d'évaluation.

Métriques calculées :
  - chrF++ (sacrebleu) : similitude lexicale FR, robuste aux variations morphologiques
  - BERTScore (optionnel, nécessite torch+transformers) : similitude sémantique
  - Ratio longueur : longueur sortie / longueur référence
  - Taux de répétition : proportion de trigrammes répétés dans la réponse
  - Langue détectée : % de réponses en français
  - Conformité structurelle : format JSON valide, alternance rôles

Usage :
  python pipeline/evaluate.py \\
      --eval-dataset training/eval-dataset.jsonl \\
      --predictions outputs/predictions-suddenly-7b.jsonl \\
      --output results/eval-suddenly-7b.json

Format de predictions.jsonl (identique au dataset d'éval, avec réponses du modèle) :
  {"messages": [...], "meta": {"genre": "...", "turns": N}}
  Les contenus assistant sont remplacés par les sorties du modèle.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Optional

try:
    from sacrebleu.metrics import CHRF
    _CHRF = CHRF(word_order=2)
    _SACREBLEU_AVAILABLE = True
except ImportError:
    _SACREBLEU_AVAILABLE = False

try:
    from bert_score import score as bertscore_fn
    _BERTSCORE_AVAILABLE = True
except ImportError:
    _BERTSCORE_AVAILABLE = False

try:
    from langdetect import detect, LangDetectException
    _LANGDETECT_AVAILABLE = True
except ImportError:
    _LANGDETECT_AVAILABLE = False


# ---------------------------------------------------------------------------
# Structures de données
# ---------------------------------------------------------------------------

@dataclass
class EvalResult:
    example_id: int
    genre: str
    turns: int
    chrf: Optional[float] = None
    bertscore_f1: Optional[float] = None
    length_ratio: Optional[float] = None
    repetition_ratio: Optional[float] = None
    lang_detected: Optional[str] = None
    structural_ok: bool = True
    error: Optional[str] = None


@dataclass
class EvalSummary:
    total: int = 0
    structural_ok: int = 0
    avg_chrf: Optional[float] = None
    avg_bertscore_f1: Optional[float] = None
    avg_length_ratio: Optional[float] = None
    avg_repetition: Optional[float] = None
    pct_french: Optional[float] = None
    by_genre: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Métriques individuelles
# ---------------------------------------------------------------------------

def _compute_chrf(hypothesis: str, reference: str) -> Optional[float]:
    if not _SACREBLEU_AVAILABLE:
        return None
    result = _CHRF.sentence_score(hypothesis, [reference])
    return round(result.score / 100.0, 4)


def _compute_repetition(text: str) -> float:
    words = text.lower().split()
    if len(words) < 3:
        return 0.0
    trigrams = [tuple(words[i:i+3]) for i in range(len(words) - 2)]
    if not trigrams:
        return 0.0
    unique = len(set(trigrams))
    return round(1.0 - unique / len(trigrams), 4)


def _compute_length_ratio(hypothesis: str, reference: str) -> float:
    ref_words = len(reference.split())
    hyp_words = len(hypothesis.split())
    if ref_words == 0:
        return 0.0
    return round(hyp_words / ref_words, 4)


def _detect_lang(text: str) -> Optional[str]:
    if not _LANGDETECT_AVAILABLE:
        return None
    try:
        return detect(text)
    except Exception:
        return None


def _is_structurally_valid(messages: list[dict]) -> bool:
    if not messages or not isinstance(messages, list):
        return False
    non_system = [m for m in messages if m.get("role") != "system"]
    if len(non_system) < 2:
        return False
    for i, msg in enumerate(non_system):
        expected = "user" if i % 2 == 0 else "assistant"
        if msg.get("role") != expected:
            return False
    return True


def _extract_assistant_texts(messages: list[dict]) -> list[str]:
    return [m["content"] for m in messages if m.get("role") == "assistant" and m.get("content")]


# ---------------------------------------------------------------------------
# Évaluation d'un exemple
# ---------------------------------------------------------------------------

def _evaluate_single(
    pred: dict,
    ref: dict,
    example_id: int,
    compute_bertscore: bool = False,
) -> EvalResult:
    meta = ref.get("meta", {})
    genre = meta.get("genre", "unknown")
    turns = meta.get("turns", 0)

    result = EvalResult(example_id=example_id, genre=genre, turns=turns)

    pred_msgs = pred.get("messages", [])
    ref_msgs = ref.get("messages", [])

    if not _is_structurally_valid(pred_msgs):
        result.structural_ok = False
        result.error = "invalid structure"
        return result

    pred_texts = _extract_assistant_texts(pred_msgs)
    ref_texts = _extract_assistant_texts(ref_msgs)

    if not pred_texts or not ref_texts:
        result.structural_ok = False
        result.error = "no assistant turns"
        return result

    # Concaténer tous les tours pour les métriques globales
    pred_concat = " ".join(pred_texts)
    ref_concat = " ".join(ref_texts)

    result.chrf = _compute_chrf(pred_concat, ref_concat)
    result.length_ratio = _compute_length_ratio(pred_concat, ref_concat)
    result.repetition_ratio = _compute_repetition(pred_concat)
    result.lang_detected = _detect_lang(pred_concat)

    return result


# ---------------------------------------------------------------------------
# Évaluation du dataset complet
# ---------------------------------------------------------------------------

def evaluate_dataset(
    predictions_path: Path,
    eval_path: Path,
    compute_bertscore: bool = False,
) -> list[EvalResult]:
    refs = []
    with eval_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                refs.append(json.loads(line))

    preds = []
    with predictions_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                preds.append(json.loads(line))

    if len(preds) != len(refs):
        print(
            f"[WARN] Nombre de prédictions ({len(preds)}) ≠ nombre de références ({len(refs)}). "
            "Évaluation sur min(preds, refs) exemples.",
            file=sys.stderr,
        )

    results = []
    n = min(len(preds), len(refs))
    for i in range(n):
        result = _evaluate_single(preds[i], refs[i], example_id=i, compute_bertscore=False)
        results.append(result)

    # BERTScore en batch (coûteux, optionnel)
    if compute_bertscore and _BERTSCORE_AVAILABLE:
        valid = [(r, preds[r.example_id], refs[r.example_id]) for r in results if r.structural_ok]
        if valid:
            hyps = [" ".join(_extract_assistant_texts(p.get("messages", []))) for _, p, _ in valid]
            refs_texts = [" ".join(_extract_assistant_texts(r.get("messages", []))) for _, _, r in valid]
            _, _, F1 = bertscore_fn(hyps, refs_texts, lang="fr", verbose=False)
            for (result, _, _), f1 in zip(valid, F1.tolist()):
                result.bertscore_f1 = round(f1, 4)
    elif compute_bertscore and not _BERTSCORE_AVAILABLE:
        print("[WARN] bert-score non installé — BERTScore ignoré.", file=sys.stderr)

    return results


# ---------------------------------------------------------------------------
# Résumé et affichage
# ---------------------------------------------------------------------------

def _compute_summary(results: list[EvalResult]) -> EvalSummary:
    summary = EvalSummary(total=len(results))
    summary.structural_ok = sum(1 for r in results if r.structural_ok)

    valid = [r for r in results if r.structural_ok]
    if not valid:
        return summary

    def _avg(values: list) -> Optional[float]:
        vals = [v for v in values if v is not None]
        return round(sum(vals) / len(vals), 4) if vals else None

    summary.avg_chrf = _avg([r.chrf for r in valid])
    summary.avg_bertscore_f1 = _avg([r.bertscore_f1 for r in valid])
    summary.avg_length_ratio = _avg([r.length_ratio for r in valid])
    summary.avg_repetition = _avg([r.repetition_ratio for r in valid])

    langs = [r.lang_detected for r in valid if r.lang_detected is not None]
    if langs:
        summary.pct_french = round(sum(1 for l in langs if l == "fr") / len(langs) * 100, 1)

    genres = sorted(set(r.genre for r in valid))
    for genre in genres:
        genre_results = [r for r in valid if r.genre == genre]
        summary.by_genre[genre] = {
            "n": len(genre_results),
            "avg_chrf": _avg([r.chrf for r in genre_results]),
            "avg_length_ratio": _avg([r.length_ratio for r in genre_results]),
            "avg_repetition": _avg([r.repetition_ratio for r in genre_results]),
        }

    return summary


def _print_summary(summary: EvalSummary) -> None:
    print(f"\n{'='*60}")
    print(f"RÉSULTATS D'ÉVALUATION")
    print(f"{'='*60}")
    print(f"Total exemples      : {summary.total}")
    print(f"Structure valide    : {summary.structural_ok}/{summary.total}")

    if summary.avg_chrf is not None:
        print(f"\nMétriques globales :")
        print(f"  chrF++            : {summary.avg_chrf:.4f}  (seuil : ≥0.25 acceptable, ≥0.40 bon)")
        if summary.avg_bertscore_f1 is not None:
            print(f"  BERTScore F1      : {summary.avg_bertscore_f1:.4f}  (seuil : ≥0.80 acceptable, ≥0.88 bon)")
        if summary.avg_length_ratio is not None:
            print(f"  Ratio longueur    : {summary.avg_length_ratio:.4f}  (cible : 0.5–2.0)")
        if summary.avg_repetition is not None:
            print(f"  Taux répétition   : {summary.avg_repetition:.4f}  (cible : <0.05)")
        if summary.pct_french is not None:
            print(f"  % réponses FR     : {summary.pct_french:.1f}%  (cible : 100%)")

    if summary.by_genre:
        print(f"\nPar genre :")
        print(f"  {'Genre':<30} {'N':>4}  {'chrF++':>7}  {'LenRatio':>9}  {'Répét.':>7}")
        print(f"  {'-'*30} {'----':>4}  {'------':>7}  {'---------':>9}  {'-------':>7}")
        for genre, stats in sorted(summary.by_genre.items()):
            chrf = f"{stats['avg_chrf']:.4f}" if stats['avg_chrf'] is not None else "  N/A "
            lr = f"{stats['avg_length_ratio']:.4f}" if stats['avg_length_ratio'] is not None else "  N/A  "
            rep = f"{stats['avg_repetition']:.4f}" if stats['avg_repetition'] is not None else "  N/A "
            print(f"  {genre:<30} {stats['n']:>4}  {chrf:>7}  {lr:>9}  {rep:>7}")

    print(f"{'='*60}\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Évalue les sorties d'un modèle contre le dataset de référence.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--eval-dataset", required=True, type=Path, help="Dataset de référence JSONL")
    parser.add_argument("--predictions", required=True, type=Path, help="Prédictions du modèle JSONL")
    parser.add_argument("--output", default=None, type=Path, help="Fichier JSON de résultats détaillés")
    parser.add_argument("--bertscore", action="store_true", help="Calculer BERTScore (nécessite torch)")
    args = parser.parse_args(argv)

    if not args.eval_dataset.exists():
        print(f"[ERROR] Dataset introuvable : {args.eval_dataset}", file=sys.stderr)
        sys.exit(1)
    if not args.predictions.exists():
        print(f"[ERROR] Prédictions introuvables : {args.predictions}", file=sys.stderr)
        sys.exit(1)

    if not _SACREBLEU_AVAILABLE:
        print("[WARN] sacrebleu non installé — chrF++ ignoré. Installer : pip install sacrebleu", file=sys.stderr)

    results = evaluate_dataset(args.predictions, args.eval_dataset, compute_bertscore=args.bertscore)
    summary = _compute_summary(results)
    _print_summary(summary)

    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "summary": asdict(summary),
            "results": [asdict(r) for r in results],
        }
        with args.output.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] Résultats détaillés écrits dans {args.output}")


if __name__ == "__main__":
    main()
