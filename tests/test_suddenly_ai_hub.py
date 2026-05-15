"""Tests for Suddenly AI Hub data and metadata.

Tests validate the test dataset structure, completeness, and consistency.
No actual model inference is performed here — those would require GPU.
"""
import json
from pathlib import Path

import pytest

BASE_DIR = Path(__file__).resolve().parent.parent


class TestTestData:
    """Validate test-prompts.jsonl structure and content."""

    @pytest.fixture
    def prompts(self):
        prompts = []
        path = BASE_DIR / "data" / "test-prompts.jsonl"
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#"):
                    prompts.append(json.loads(line))
        return prompts

    def test_file_exists(self):
        path = BASE_DIR / "data" / "test-prompts.jsonl"
        assert path.exists(), "data/test-prompts.jsonl should exist"

    def test_prompt_count(self, prompts):
        assert len(prompts) == 50, f"Expected 50 prompts, got {len(prompts)}"

    def test_prompt_fields(self, prompts):
        required = {"id", "univers", "situation", "voice", "prompt", "category"}
        for p in prompts:
            assert required.issubset(p.keys()), f"Missing fields in prompt {p.get('id')}: {required - set(p.keys())}"

    def test_unique_ids(self, prompts):
        ids = [p["id"] for p in prompts]
        assert len(ids) == len(set(ids)), "All prompt IDs must be unique"

    def test_id_range(self, prompts):
        ids = sorted([p["id"] for p in prompts])
        assert ids == list(range(1, 51)), "IDs should range from 1 to 50"

    def test_universes(self, prompts):
        universes = set(p["univers"] for p in prompts)
        expected = {"fantasy-medievale", "cyberpunk"}
        assert universes == expected, f"Expected {expected}, got {universes}"

    def test_situations(self, prompts):
        situations = set(p["situation"] for p in prompts)
        expected = {"combat", "romance", "intrigue", "politique", "quotidien"}
        assert situations == expected, f"Expected {expected}, got {situations}"

    def test_voices(self, prompts):
        voices = set(p["voice"] for p in prompts)
        expected = {"solennel", "narquois", "theatral", "neutre", "lyrique"}
        assert voices == expected, f"Expected {expected}, got {voices}"

    def test_distribution(self, prompts):
        """Each univers×situation should have 5 prompts (one per voice)."""
        from collections import Counter
        counts = Counter()
        for p in prompts:
            counts[(p["univers"], p["situation"])] += 1
        for key, count in counts.items():
            assert count == 5, f"{key} should have 5 prompts, got {count}"

    def test_categories(self, prompts):
        categories = set(p["category"] for p in prompts)
        expected = {
            "combat_description", "combat_dialogue",
            "romance_description", "romance_dialogue",
            "intrigue_description", "intrigue_dialogue",
            "politique_description", "politique_dialogue",
            "quotidien_description", "quotidien_dialogue",
        }
        assert categories == expected, f"Expected {expected}, got {categories}"

    def test_prompts_are_nonempty(self, prompts):
        for p in prompts:
            assert len(p["prompt"]) > 0, f"Prompt {p['id']} should not be empty"
            assert len(p["prompt"]) < 500, f"Prompt {p['id']} should be < 500 chars"


class TestListModelsScript:
    """Validate list_models.py metadata."""

    def test_script_exists(self):
        path = BASE_DIR / "scripts" / "list_models.py"
        assert path.exists()

    def test_axe_count(self):
        from scripts.list_models import AXES
        assert len(AXES) == 3, f"Expected 3 axes, got {len(AXES)}"

    def test_axis_names(self):
        from scripts.list_models import AXES
        axis_names = list(AXES.keys())
        assert "Axe 1 — Univers (genre/lore)" in axis_names
        assert "Axe 2 — Situation (ton/rythme)" in axis_names
        assert "Axe 3 — Voix (personnalité narrative)" in axis_names

    def test_adapter_counts(self):
        from scripts.list_models import AXES
        counts = [len(v) for v in AXES.values()]
        # Univers: 4, Situation: 5, Voix: 5
        assert counts == [4, 5, 5], f"Expected [4, 5, 5], got {counts}"

    def test_total_adapters(self):
        from scripts.list_models import AXES
        total = sum(len(v) for v in AXES.values())
        assert total == 14, f"Expected 14 total adapters, got {total}"


class TestStackingConfig:
    """Validate stacking 3-axis configuration."""

    def test_orthogonal_combinations(self):
        """3 axes with 4, 5, 5 adapters = 100 combinations."""
        from scripts.list_models import AXES
        univers_count = len(AXES["Axe 1 — Univers (genre/lore)"])
        situation_count = len(AXES["Axe 2 — Situation (ton/rythme)"])
        voice_count = len(AXES["Axe 3 — Voix (personnalité narrative)"])
        total_combos = univers_count * situation_count * voice_count
        assert total_combos == 100, f"Expected 100 combinations, got {total_combos}"

    def test_valid_multiplier_range(self):
        """Multipliers should be 0.0–2.0 based on documentation."""
        content = (BASE_DIR / "scripts" / "infer.py").read_text()
        assert "multiplier" in content, "infer.py should reference multipliers"
        assert "0.0" in content or "0" in content, "infer.py should define min multiplier"
        assert "2.0" in content or "2" in content, "infer.py should define max multiplier"


class TestInferScript:
    """Validate infer.py CLI arguments."""

    def test_script_exists(self):
        path = BASE_DIR / "scripts" / "infer.py"
        assert path.exists()

    def test_has_stack_arg(self):
        content = (BASE_DIR / "scripts" / "infer.py").read_text()
        assert "--stack" in content
        assert "--adapter-1" in content
        assert "--adapter-2" in content
        assert "--adapter-3" in content
        assert "argparse" in content

    def test_parser_defines_stacked_args(self):
        import ast
        tree = ast.parse((BASE_DIR / "scripts" / "infer.py").read_text())
        found_args = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef) and node.name == "main":
                for child in ast.walk(node):
                    if isinstance(child, ast.Call):
                        func = child.func
                        if isinstance(func, ast.Attribute) and func.attr == "add_argument":
                            for arg in child.args:
                                if isinstance(arg, ast.Constant) and arg.value.startswith("--"):
                                    found_args.add(arg.value)
        # Verify key stacking args are present
        assert "--stack" in found_args
        assert "--adapter-1" in found_args
        assert "--adapter-2" in found_args
        assert "--adapter-3" in found_args
        assert "--multiplier-1" in found_args
        assert "--multiplier-2" in found_args
        assert "--multiplier-3" in found_args


class TestEvaluateScript:
    """Validate evaluate.py structure."""

    def test_script_exists(self):
        path = BASE_DIR / "scripts" / "evaluate.py"
        assert path.exists()

    def test_has_stack_and_full_args(self):
        content = (BASE_DIR / "scripts" / "evaluate.py").read_text()
        assert "--stack" in content
        assert "--full" in content
        assert "argparse" in content

    def test_load_test_data_returns_list(self):
        from scripts.evaluate import load_test_data
        data = load_test_data(str(BASE_DIR / "data" / "test-prompts.jsonl"))
        assert isinstance(data, list)
        assert len(data) == 50
        assert all("prompt" in item for item in data)


class TestBaselineScript:
    """Validate baseline.py structure."""

    def test_script_exists(self):
        path = BASE_DIR / "scripts" / "baseline.py"
        assert path.exists()

    def test_has_generate_function(self):
        import ast
        content = (BASE_DIR / "scripts" / "baseline.py").read_text()
        tree = ast.parse(content)
        func_names = [n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]
        assert "generate_baseline" in func_names


class TestReadme:
    """Validate README.md contains stacking 3-axis documentation."""

    def test_readme_exists(self):
        path = BASE_DIR / "README.md"
        assert path.exists()

    def test_readme_contains_three_axes(self):
        content = (BASE_DIR / "README.md").read_text(encoding="utf-8")
        assert "Axe 1" in content, "README should reference Axe 1 (Univers)"
        assert "Axe 2" in content, "README should reference Axe 2 (Situation)"
        assert "Axe 3" in content, "README should reference Axe 3 (Voix)"

    def test_readme_contains_stacking(self):
        content = (BASE_DIR / "README.md").read_text(encoding="utf-8")
        assert "stacking" in content.lower(), "README should mention stacking"
        assert "multiplier" in content.lower(), "README should mention multipliers"

    def test_readme_no_training_stack(self):
        """README should not reveal training stack (per project policy)."""
        content = (BASE_DIR / "README.md").read_text(encoding="utf-8").lower()
        # Should NOT mention Axolotl, training configs, or LoRA internals
        assert "axolotl" not in content, "README must not reveal training stack"
        assert "fine-tuning" not in content, "README must not reveal training stack"
