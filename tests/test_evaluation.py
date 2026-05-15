"""Unit tests for TDD evaluation scoring — criteria-based PASS/FAIL."""
import json
import sys
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))

from evaluate import (
    extract_criteria_score,
    score_output,
    load_test_data,
    compute_category_scores,
    run_compare,
    generate_mock_response,
)


class TestExtractCriteriaScore:
    """Test keyword-based criteria extraction."""

    def test_univers_match(self):
        output = "Le chevalier brandit son épée dans la lumière dorée du soleil couchant."
        criteria = {
            "univers": ["épée", "lumière", "soleil"],
            "situation": ["attaque", "combat", "lame"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert "épée" in details["univers"]["matched"]
        assert "lumière" in details["univers"]["matched"]
        assert "soleil" in details["univers"]["matched"]
        assert details["univers"]["pass"] is True

    def test_situation_partial_match(self):
        """Only 1/3 keywords match — should pass (>= 33% threshold)."""
        output = "Le chevalier lance son attaque finale."
        criteria = {
            "univers": ["épée", "lumière", "soleil"],
            "situation": ["attaque", "combat", "lame"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["situation"]["pass"] is True
        assert len(details["situation"]["matched"]) >= 1

    def test_situation_no_match(self):
        """0/3 keywords match — should fail."""
        output = "Le ciel est bleu aujourd'hui."
        criteria = {
            "univers": ["épée", "lumière", "soleil"],
            "situation": ["attaque", "combat", "lame"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["situation"]["pass"] is False

    def test_voix_match(self):
        output = "D'une voix solennelle et majestueuse, le chevalier avance."
        criteria = {
            "univers": ["épée", "lumière", "soleil"],
            "situation": ["attaque", "combat", "lame"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["voix"]["pass"] is True

    def test_langue_french(self):
        """Texte en français détecté grâce aux mots communs."""
        output = "Le chevalier brandit son épée dans la lumière."
        criteria = {
            "univers": ["épée"],
            "situation": ["combat"],
            "voix": ["solennel"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["langue"]["pass"] is True

    def test_langue_non_french(self):
        """Texte en anglais ne doit pas passer le critère langue."""
        output = "The knight draws his sword in the light."
        criteria = {
            "univers": ["épée"],
            "situation": ["combat"],
            "voix": ["solennel"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["langue"]["pass"] is False

    def test_all_dimensions_pass(self):
        """If ALL dimensions pass, overall should be True."""
        output = "Le vieux chevalier lance son attaque avec épée, la lame de lumière au soleil, d'une voix solennelle."
        criteria = {
            "univers": ["épée", "lumière", "soleil"],
            "situation": ["attaque", "combat"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert passed is True  # All dims pass → overall True

    def test_one_dimension_fails(self):
        """If ANY dimension fails, overall should be False."""
        output = "The knight draws his sword in the light."
        criteria = {
            "univers": ["épée", "lumière", "soleil"],
            "situation": ["attaque", "combat"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert passed is False  # langue + univers should fail

    def test_empty_criteria(self):
        """Empty criteria dict — no dimension check possible → FAIL (reject, don't auto-pass)."""
        output = "test"
        passed, details = extract_criteria_score(output, {})
        assert passed is False  # No criteria = reject, not auto-pass

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        output = "LE CHEVALIER BRANDIT SON ÉPÉE"
        criteria = {
            "univers": ["épée", "lumière"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["univers"]["pass"] is True
        assert "épée" in details["univers"]["matched"]


class TestScoreOutput:
    """Test the full score_output function."""

    def test_score_with_criteria(self):
        output = "Le chevalier brandit son épée dans la lumière dorée."
        prompt_data = {
            "criteria": {
                "univers": ["épée", "lumière", "soleil"],
                "situation": ["attaque"],
                "voix": ["solennel"],
                "langue": ["fr"],
            }
        }
        result = score_output(output, prompt_data)
        assert "pass" in result
        assert "details" in result
        assert "scores" in result
        assert "univers" in result["scores"]
        assert "situation" in result["scores"]
        assert "voix" in result["scores"]
        assert "langue" in result["scores"]

    def test_score_without_criteria(self):
        """No criteria → fallback return."""
        output = "test output"
        prompt_data = {"prompt": "test"}
        result = score_output(output, prompt_data)
        assert result["pass"] is False
        assert result["reason"] == "no_criteria"

    def test_score_returns_float_values(self):
        """Scores should be float 0.0-1.0 (capped)."""
        output = "Le chevalier brandit son épée."
        prompt_data = {
            "criteria": {
                "univers": ["épée"],
                "situation": ["combat"],
                "voix": ["solennel"],
                "langue": ["fr"],
            }
        }
        result = score_output(output, prompt_data)
        for dim, score in result["scores"].items():
            assert isinstance(score, float) or isinstance(score, int)
            assert 0.0 <= score <= 1.0  # Scores are now capped at 1.0


class TestComputeCategoryScores:
    """Test score aggregation."""

    def test_empty_list(self):
        result = compute_category_scores([])
        assert result == {}

    def test_single_result(self):
        results = [
            {
                "score": {
                    "pass": True,
                    "scores": {"univers": 1.0, "situation": 0.5, "voix": 1.0, "langue": 1.0},
                }
            }
        ]
        agg = compute_category_scores(results)
        assert agg["count"] == 1
        assert agg["passed"] == 1
        assert agg["failed"] == 0
        assert agg["pass_rate"] == 1.0

    def test_mixed_results(self):
        results = [
            {"score": {"pass": True, "scores": {"univers": 1.0, "situation": 1.0, "voix": 1.0, "langue": 1.0}}},
            {"score": {"pass": False, "scores": {"univers": 0.33, "situation": 0.0, "voix": 0.0, "langue": 1.0}}},
            {"score": {"pass": True, "scores": {"univers": 1.0, "situation": 1.0, "voix": 0.66, "langue": 1.0}}},
        ]
        agg = compute_category_scores(results)
        assert agg["count"] == 3
        assert agg["passed"] == 2
        assert agg["failed"] == 1
        assert agg["pass_rate"] == 2 / 3

    def test_avg_scores_computed(self):
        results = [
            {"score": {"pass": True, "scores": {"univers": 1.0, "situation": 0.5, "voix": 0.5, "langue": 1.0}}},
            {"score": {"pass": False, "scores": {"univers": 0.0, "situation": 0.0, "voix": 0.0, "langue": 1.0}}},
        ]
        agg = compute_category_scores(results)
        assert "avg_scores" in agg
        # univers avg = (1.0 + 0.0) / 2 = 0.5
        assert agg["avg_scores"]["univers"] == 0.5
        # situation avg = (0.5 + 0.0) / 2 = 0.25
        assert agg["avg_scores"]["situation"] == 0.25


class TestLoadTestData:
    """Test loading test prompts."""

    def test_load_prompts(self, tmp_path):
        data_file = tmp_path / "test-prompts.jsonl"
        test_prompts = [
            {"id": 1, "univers": "test", "criteria": {"univers": ["kw1"], "langue": ["fr"]}},
            {"id": 2, "univers": "test2", "criteria": {"univers": ["kw2"], "langue": ["fr"]}},
        ]
        with open(data_file, "w") as f:
            for p in test_prompts:
                f.write(json.dumps(p) + "\n")

        loaded = load_test_data(str(data_file))
        assert len(loaded) == 2
        assert loaded[0]["id"] == 1
        assert loaded[1]["criteria"]["univers"] == ["kw2"]

    def test_load_with_empty_lines(self, tmp_path):
        data_file = tmp_path / "test-prompts.jsonl"
        with open(data_file, "w") as f:
            f.write('{"id": 1, "criteria": {"univers": ["a"], "langue": ["fr"]}}\n')
            f.write('\n')
            f.write('{"id": 2, "criteria": {"univers": ["b"], "langue": ["fr"]}}\n')

        loaded = load_test_data(str(data_file))
        assert len(loaded) == 2


class TestBaselineExpectedFail:
    """TDD test: baseline model output should FAIL criteria (simulated)."""

    def test_generic_output_fails_criteria(self):
        """A generic/non-RP output should fail univers/situation/voix criteria."""
        generic_output = "Here is some information about the topic. The answer is provided above."
        criteria = {
            "univers": ["épée", "lumière", "royaume"],
            "situation": ["combat", "attaque", "lame"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(generic_output, criteria)
        assert passed is False  # Baseline SHOULD fail — this is the TDD "red" phase

    def test_rich_rp_output_passes_criteria(self):
        """A rich RP output should pass ALL criteria."""
        rp_output = "D'une voix solennelle, le vieux chevalier brandit son épée, la lame de lumière jaillit au coucher du soleil dans le royaume."
        criteria = {
            "univers": ["épée", "lumière", "royaume"],
            "situation": ["combat", "attaque", "lame"],
            "voix": ["solennel", "grave", "majestueux"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(rp_output, criteria)
        assert passed is True  # Fine-tuned SHOULD pass — this is the TDD "green" phase


class TestNarquoisVoice:
    """Test narquois (sarcastic) voice criteria."""

    def test_narquois_keywords(self):
        output = "Le marchand cynique lance une réplique ironique, pince-sans-rire."
        criteria = {
            "univers": ["épée", "marchand"],
            "situation": ["combat", "réplique"],
            "voix": ["cynique", "ironique", "pince-sans-rire"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["voix"]["pass"] is True
        assert len(details["voix"]["matched"]) >= 1  # At least 1/3

    def test_lyrique_keywords(self):
        output = "Les lames dansent de manière poétique, une métaphore de mort et de beauté."
        criteria = {
            "univers": ["champions", "arène", "épée"],
            "situation": ["duel", "combat"],
            "voix": ["poétique", "sensualité", "métaphore"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["voix"]["pass"] is True
        assert "poétique" in details["voix"]["matched"]
        assert "métaphore" in details["voix"]["matched"]


class TestCyberpunkWorld:
    """Test cyberpunk universe criteria."""

    def test_cyberpunk_keywords(self):
        output = "Dans Neo-Tokyo, les néons éclairent la mégapole tandis que le hacker utilise son virus."
        criteria = {
            "univers": ["hacker", "virus", "néons"],
            "situation": ["neutralise", "gardes", "séquence"],
            "voix": ["espiègle", "humour", "ironique"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["univers"]["pass"] is True
        assert "néons" in details["univers"]["matched"]
        assert "hacker" in details["univers"]["matched"]

    def test_cyberpunk_no_fantasy(self):
        """Cyberpunk output should not match fantasy criteria."""
        output = "Le chevalier brandit son épée dans le royaume enchanté."
        criteria = {
            "univers": ["hacker", "virus", "néons"],
            "situation": ["combat"],
            "voix": ["solennel"],
            "langue": ["fr"],
        }
        passed, details = extract_criteria_score(output, criteria)
        assert details["univers"]["pass"] is False  # No cyberpunk keywords


class TestLangueScoreNormalization:
    """Test that langue scores are properly normalized and capped."""

    def test_langue_score_normalization(self):
        """Even with many French words, score should be normalized."""
        # Text with many common French words
        output = "Le chat noir est sur la table en attendant de manger avec de la soupe dans un bol."
        prompt_data = {
            "criteria": {
                "univers": ["chat"],
                "langue": ["fr"],
            }
        }
        result = score_output(output, prompt_data)
        assert "langue" in result["scores"]
        # ~14 common French words found, normalized to 14/27 ≈ 0.52, not > 1.0
        assert 0.0 <= result["scores"]["langue"] <= 1.0

    def test_langue_score_capped_at_one(self):
        """Score should not exceed 1.0 even with many matches."""
        # Text with lots of common French words
        output = "Le la les de du des et est dans avec sur pour que une ce son sa ses un il elle ils elles au aux en ne pas plus tout comme le la les de du des et est dans avec sur."
        prompt_data = {
            "criteria": {
                "univers": ["mot"],
                "langue": ["fr"],
            }
        }
        result = score_output(output, prompt_data)
        assert result["scores"]["langue"] <= 1.0


if __name__ == "__main__":
    import pytest
    pytest.main([__file__, "-v"])


class TestGenerateMockResponse:
    """Test mock response generation for TDD."""

    def test_mock_contains_all_keywords(self):
        """The mock should include ALL keywords from ALL criteria."""
        prompt_data = {
            "criteria": {
                "univers": ["épée", "lumière", "royaume"],
                "situation": ["combat", "attaque"],
                "voix": ["solennel", "grave"],
                "langue": ["fr"],
            }
        }
        output = generate_mock_response(prompt_data)
        # Univers keywords
        assert "épée" in output
        assert "lumière" in output
        assert "royaume" in output
        # Situation keywords
        assert "combat" in output
        assert "attaque" in output
        # Voix keywords
        assert "solennel" in output
        assert "grave" in output
        # langue keywords should NOT be included
        assert "fr" not in output.split()

    def test_mock_contains_french_common_words(self):
        """Mock should include 'le', 'la', 'dans' to pass langue criterion."""
        prompt_data = {
            "criteria": {
                "univers": ["épée"],
                "situation": ["combat"],
                "voix": ["solennel"],
                "langue": ["fr"],
            }
        }
        output = generate_mock_response(prompt_data)
        assert "le" in output
        assert "la" in output
        assert "dans" in output

    def test_mock_with_empty_criteria(self):
        """Empty criteria → should return just 'le la dans'."""
        prompt_data = {"criteria": {}}
        output = generate_mock_response(prompt_data)
        assert output == "le la dans"

    def test_mock_deterministic(self):
        """Same prompt_data → same output."""
        prompt_data = {
            "criteria": {
                "univers": ["dragon"],
                "situation": ["vol"],
                "voix": ["terrible"],
                "langue": ["fr"],
            }
        }
        output1 = generate_mock_response(prompt_data)
        output2 = generate_mock_response(prompt_data)
        assert output1 == output2


class TestMockPassesAllPrompts:
    """Integration test: mock must pass ALL 50 real prompts."""

    def test_mock_passes_all_real_prompts(self):
        """All 50 prompts from test-prompts.jsonl should PASS with mock."""
        test_data = load_test_data("data/test-prompts.jsonl")
        assert len(test_data) == 50

        passed = 0
        for data in test_data:
            output = generate_mock_response(data)
            result = score_output(output, data)
            if result["pass"]:
                passed += 1
            else:
                # Debug: show which prompt failed
                print(f"\n  FAILED: {data.get('id')} - {data.get('category', '?')} "
                      f"({data.get('univers', '?')})")
                print(f"    Output: {output[:80]}")
                print(f"    Criteria: {data.get('criteria', {})}")
                print(f"    Score: {result}")

        assert passed == 50, f"Only {passed}/50 prompts passed mock evaluation"

    def test_mock_scores_are_valid_floats(self):
        """All mock score values should be valid floats 0.0-1.0."""
        test_data = load_test_data("data/test-prompts.jsonl")
        for data in test_data[:10]:  # Sample 10
            output = generate_mock_response(data)
            result = score_output(output, data)
            for dim, score in result["scores"].items():
                assert 0.0 <= score <= 1.0, f"Score {dim}={score} out of range"


class TestMockVsBaselineCompare:
    """Test mock vs baseline comparison flow."""

    def test_run_compare_with_mock_results(self):
        """run_compare should work when called with mock results."""
        # Simulate mock results (all pass)
        test_data = load_test_data("data/test-prompts.jsonl")
        mock_results = []
        for data in test_data[:5]:  # Sample 5 for speed
            output = generate_mock_response(data)
            score = score_output(output, data)
            mock_results.append({
                "id": data["id"],
                "prompt": data["prompt"],
                "output": output,
                "criteria": data.get("criteria", {}),
                "score": score,
                "metadata": {
                    "univers": data.get("univers", ""),
                    "situation": data.get("situation", ""),
                    "voice": data.get("voice", ""),
                    "category": data.get("category", ""),
                }
            })

        # Simulate baseline results (all fail)
        baseline_results = []
        for data in test_data[:5]:
            score = score_output("generic baseline output", data)
            baseline_results.append({
                "id": data["id"],
                "prompt": data["prompt"],
                "output": "generic baseline output",
                "criteria": data.get("criteria", {}),
                "score": score,
                "metadata": {
                    "univers": data.get("univers", ""),
                    "situation": data.get("situation", ""),
                    "voice": data.get("voice", ""),
                    "category": data.get("category", ""),
                }
            })

        # run_compare should execute without error
        # (it prints to stdout, we just verify it doesn't crash)
        run_compare(baseline_results, mock_results)

        # Verify baseline failed, mock passed
        b_passed = sum(1 for r in baseline_results if r["score"]["pass"])
        m_passed = sum(1 for r in mock_results if r["score"]["pass"])
        assert b_passed < m_passed, "Baseline should have fewer passes than mock"
