import json
from pathlib import Path
from unittest.mock import patch

import pytest

from pipeline.evaluate import (
    _compute_length_ratio,
    _compute_repetition,
    _compute_summary,
    _evaluate_single,
    _is_structurally_valid,
    EvalResult,
    evaluate_dataset,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_ref(genre: str = "generique", turns: int = 1, assistant_text: str = "réponse de référence") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Contexte RP."},
            {"role": "user", "content": "action du joueur"},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": {"genre": genre, "turns": turns},
    }


def _make_pred(assistant_text: str = "réponse du modèle") -> dict:
    return {
        "messages": [
            {"role": "system", "content": "Contexte RP."},
            {"role": "user", "content": "action du joueur"},
            {"role": "assistant", "content": assistant_text},
        ],
        "meta": {"genre": "generique", "turns": 1},
    }


# ---------------------------------------------------------------------------
# _is_structurally_valid
# ---------------------------------------------------------------------------

class TestIsStructurallyValid:
    def test_valid(self):
        msgs = [
            {"role": "user", "content": "action"},
            {"role": "assistant", "content": "réponse"},
        ]
        assert _is_structurally_valid(msgs) is True

    def test_valid_with_system(self):
        msgs = [
            {"role": "system", "content": "ctx"},
            {"role": "user", "content": "action"},
            {"role": "assistant", "content": "réponse"},
        ]
        assert _is_structurally_valid(msgs) is True

    def test_empty(self):
        assert _is_structurally_valid([]) is False

    def test_only_system(self):
        assert _is_structurally_valid([{"role": "system", "content": "x"}]) is False

    def test_wrong_order(self):
        msgs = [
            {"role": "assistant", "content": "réponse"},
            {"role": "user", "content": "action"},
        ]
        assert _is_structurally_valid(msgs) is False

    def test_single_user(self):
        assert _is_structurally_valid([{"role": "user", "content": "x"}]) is False


# ---------------------------------------------------------------------------
# _compute_repetition
# ---------------------------------------------------------------------------

class TestComputeRepetition:
    def test_no_repetition(self):
        text = "la forêt bruissait sous le vent froid de l'automne"
        ratio = _compute_repetition(text)
        assert ratio == 0.0

    def test_full_repetition(self):
        text = "a b c a b c a b c a b c"
        ratio = _compute_repetition(text)
        assert ratio > 0.5

    def test_short_text(self):
        assert _compute_repetition("un deux") == 0.0

    def test_empty(self):
        assert _compute_repetition("") == 0.0


# ---------------------------------------------------------------------------
# _compute_length_ratio
# ---------------------------------------------------------------------------

class TestComputeLengthRatio:
    def test_equal(self):
        assert _compute_length_ratio("un deux trois", "un deux trois") == 1.0

    def test_double(self):
        ratio = _compute_length_ratio("un deux trois quatre six sept", "un deux trois")
        assert abs(ratio - 2.0) < 0.1

    def test_half(self):
        ratio = _compute_length_ratio("un", "un deux")
        assert abs(ratio - 0.5) < 0.1

    def test_empty_reference(self):
        assert _compute_length_ratio("quelque chose", "") == 0.0


# ---------------------------------------------------------------------------
# _evaluate_single
# ---------------------------------------------------------------------------

class TestEvaluateSingle:
    def test_valid_example_returns_result(self):
        ref = _make_ref(genre="medieval-fantastique", turns=1, assistant_text="texte de référence long")
        pred = _make_pred("texte de prédiction du modèle")
        result = _evaluate_single(pred, ref, example_id=0)
        assert result.structural_ok is True
        assert result.genre == "medieval-fantastique"
        assert result.turns == 1
        assert result.length_ratio is not None
        assert result.repetition_ratio is not None

    def test_invalid_structure_flagged(self):
        ref = _make_ref()
        pred = {"messages": [{"role": "assistant", "content": "sans user"}], "meta": {}}
        result = _evaluate_single(pred, ref, example_id=0)
        assert result.structural_ok is False
        assert result.error is not None

    def test_empty_messages_flagged(self):
        ref = _make_ref()
        pred = {"messages": [], "meta": {}}
        result = _evaluate_single(pred, ref, example_id=0)
        assert result.structural_ok is False

    def test_meta_extracted(self):
        ref = _make_ref(genre="cyberpunk", turns=2)
        pred = _make_pred()
        result = _evaluate_single(pred, ref, example_id=5)
        assert result.example_id == 5
        assert result.genre == "cyberpunk"
        assert result.turns == 2


# ---------------------------------------------------------------------------
# evaluate_dataset — intégration
# ---------------------------------------------------------------------------

class TestEvaluateDataset:
    def test_basic_run(self, tmp_path: Path):
        eval_f = tmp_path / "eval.jsonl"
        pred_f = tmp_path / "pred.jsonl"

        refs = [_make_ref(genre=f"genre{i}", turns=1, assistant_text=f"référence {i}") for i in range(3)]
        preds = [_make_pred(f"prédiction {i}") for i in range(3)]

        eval_f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in refs), encoding="utf-8")
        pred_f.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds), encoding="utf-8")

        results = evaluate_dataset(pred_f, eval_f)
        assert len(results) == 3
        assert all(r.structural_ok for r in results)

    def test_mismatched_length_uses_min(self, tmp_path: Path):
        eval_f = tmp_path / "eval.jsonl"
        pred_f = tmp_path / "pred.jsonl"

        refs = [_make_ref() for _ in range(5)]
        preds = [_make_pred() for _ in range(3)]

        eval_f.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in refs), encoding="utf-8")
        pred_f.write_text("\n".join(json.dumps(p, ensure_ascii=False) for p in preds), encoding="utf-8")

        results = evaluate_dataset(pred_f, eval_f)
        assert len(results) == 3

    def test_output_json_written(self, tmp_path: Path):
        from pipeline.evaluate import main

        eval_f = tmp_path / "eval.jsonl"
        pred_f = tmp_path / "pred.jsonl"
        out_f = tmp_path / "results.json"

        ref = _make_ref()
        pred = _make_pred()
        eval_f.write_text(json.dumps(ref, ensure_ascii=False), encoding="utf-8")
        pred_f.write_text(json.dumps(pred, ensure_ascii=False), encoding="utf-8")

        main([
            "--eval-dataset", str(eval_f),
            "--predictions", str(pred_f),
            "--output", str(out_f),
        ])

        assert out_f.exists()
        payload = json.loads(out_f.read_text(encoding="utf-8"))
        assert "summary" in payload
        assert "results" in payload


# ---------------------------------------------------------------------------
# _compute_summary
# ---------------------------------------------------------------------------

class TestComputeSummary:
    def test_empty_results(self):
        summary = _compute_summary([])
        assert summary.total == 0
        assert summary.avg_chrf is None

    def test_all_invalid(self):
        results = [EvalResult(example_id=i, genre="g", turns=1, structural_ok=False) for i in range(3)]
        summary = _compute_summary(results)
        assert summary.structural_ok == 0
        assert summary.avg_chrf is None

    def test_by_genre(self):
        results = [
            EvalResult(example_id=0, genre="cyberpunk", turns=1, chrf=0.3, length_ratio=1.0, repetition_ratio=0.0),
            EvalResult(example_id=1, genre="cyberpunk", turns=1, chrf=0.4, length_ratio=1.2, repetition_ratio=0.0),
            EvalResult(example_id=2, genre="scifi", turns=1, chrf=0.5, length_ratio=0.9, repetition_ratio=0.0),
        ]
        summary = _compute_summary(results)
        assert "cyberpunk" in summary.by_genre
        assert "scifi" in summary.by_genre
        assert summary.by_genre["cyberpunk"]["n"] == 2
