# Plan: Mock Evaluation TDD Mode

**Date:** 2026-05-15
**Issue:** #57 — Créer les tests de validation LoRA par genre et situation
**Statut:** ✅ COMPLÉTÉ

## Problème identifié

L'issue #57 demandait des tests de validation pour les LoRA, mais l'implémentation existante (evaluate_lora.py) utilisait un scoring 1-5 par dimension incompatible avec le système de critère PASS/FAIL de evaluate.py. De plus, pas de moyen de tester le pipeline sans GPU — impossible de valider la logique de scoring en local.

## Solution implémentée

### 1. `generate_mock_response()` — evaluate.py:208-227

Injection de mots-clés pour garantir le PASS :
- Récupère tous les mots-clés de tous les critères (univers, situation, voix)
- Ignore le critère "langue" (keywords=['fr'] n'est pas un vrai mot-clé)
- Ajoute `["le", "la", "dans"]` pour passer le check "langue" (3+ mots français communs)
- **Déterministe** : toujours le même output pour un prompt donné (pas de random)

```python
def generate_mock_response(prompt_data: dict) -> str:
    criteria = prompt_data.get("criteria", {})
    words = []
    for dim, kws in criteria.items():
        if dim != "langue" and isinstance(kws, list):
            words.extend(kws)
    words.extend(["le", "la", "dans"])
    return " ".join(words) if words else "le la dans"
```

### 2. `--mock` flag — evaluate.py:430-431

```python
parser.add_argument("--mock", action="store_true",
                   help="Mock evaluation using keyword injection (TDD: should PASS)")
```

### 3. Mock evaluation loop — evaluate.py:470-507

- Parcourt les 50 prompts, appelle `generate_mock_response()` puis `score_output()`
- Affiche les résultats comme un vrai mode d'évaluation
- Si `--compare` : auto-lance baseline si pas de baseline explicite
- Early return après mock (ne continue pas vers `--stack`)

### 4. `--compare` mis à jour — evaluate.py:531-554

- Si `baseline_results is None` : auto-lance baseline
- Si `mock_results is not None` : compare baseline vs mock
- Si `--compare --baseline` sans mock/stack : affiche message instructif

## Tests unitaires ajoutés

3 nouvelles classes, 7 nouveaux tests dans `tests/test_evaluation.py` :

| Test | Vérifie |
|------|---------|
| `test_mock_contains_all_keywords` | Tous les mots-clés injectés |
| `test_mock_contains_french_common_words` | "le la dans" pour langue |
| `test_mock_with_empty_criteria` | Empty → "le la dans" |
| `test_mock_deterministic` | Même input → même output |
| `test_mock_passes_all_real_prompts` | 50/50 prompts réels PASS |
| `test_mock_scores_are_valid_floats` | Scores 0.0-1.0 |
| `test_run_compare_with_mock_results` | run_compare mock vs baseline |

**Résultat : 34/34 tests passing** (27 existants + 7 nouveaux)

## Files modifiés

| File | Lignes | Changement |
|------|--------|------------|
| `scripts/evaluate.py` | +52 | generate_mock_response() + --mock flag + mock loop + compare updates |
| `tests/test_evaluation.py` | +99 | 3 classes, 7 tests |

## Usage

```bash
# TDD mock: should PASS 50/50
python3 scripts/evaluate.py --mock

# Mock with baseline comparison
python3 scripts/evaluate.py --mock --compare

# Baseline only: should FAIL
python3 scripts/evaluate.py --baseline

# Full TDD flow: baseline (FAIL) → compare with mock (PASS)
python3 scripts/evaluate.py --baseline --mock --compare
```
