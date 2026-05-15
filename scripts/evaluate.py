#!/usr/bin/env python3
"""Evaluation pipeline for Suddenly AI Hub stacked LoRA models.

TDD Strategy: baseline (modèle brut) doit échouer sur les critères RP,
le modèle fine-tuned doit passer.

Usage:
  python scripts/evaluate.py --baseline --test-data data/test-prompts.jsonl
  python scripts/evaluate.py --stack --compare \
    --adapter-1 fantasy-medievale --multiplier-1 1.0 \
    --adapter-2 combat            --multiplier-2 1.0 \
    --adapter-3 narquois          --multiplier-3 1.0

Each prompt has criteria with keyword lists for PASS/FAIL scoring.
"""
import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def load_test_data(path: str) -> list[dict]:
    """Load JSONL test prompts with criteria."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def load_model_single(adapter_path: str, base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load a single LoRA adapter."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def load_model_stacked(
    adapter1: str, adapter2: str, adapter3: str,
    mult1: float, mult2: float, mult3: float,
    base_model: str = "Qwen/Qwen2.5-7B-Instruct",
):
    """Load stacked 3-axis adapters."""
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )

    adapters = [
        (adapter1, mult1, "univers"),
        (adapter2, mult2, "situation"),
        (adapter3, mult3, "voix"),
    ]
    for adapter_id, mult, axis_name in adapters:
        if mult == 0.0:
            continue
        adapter_path = Path(__file__).resolve().parent.parent / "models" / adapter_id
        if adapter_path.exists():
            model = PeftModel.from_pretrained(
                model, str(adapter_path), adapter_name=axis_name
            )
    model.set_adapter(["univers", "situation", "voix"])
    return model, tokenizer


def load_base_model(base_model: str = "Qwen/Qwen2.5-7B-Instruct"):
    """Load the base model without any LoRA adapters."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype="auto", device_map="auto"
    )
    return model, tokenizer


def generate_answer(
    model, tokenizer, prompt: str, max_new_tokens: int = 300, temperature: float = 0.7
) -> str:
    """Generate a response from the model."""
    full_prompt = f"👤user\n{prompt}\n\n🤖assistant\n"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
    )
    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    if "🤖assistant\n" in result:
        return result.split("🤖assistant\n")[-1].strip()
    return result.strip()


def extract_criteria_score(output: str, criteria: dict) -> tuple[bool, dict]:
    """Score an output against criteria keywords (PASS/FAIL per dimension).

    Returns (passed: bool, details: dict) where passed is True only if
    ALL dimensions have at least one keyword match.
    """
    output_lower = output.lower()
    details = {}
    all_passed = True

    for dimension, keywords in criteria.items():
        if dimension == "langue":
            # Simple French detection: common French words
            french_words = {"le", "la", "les", "de", "du", "des", "et", "est",
                           "dans", "avec", "sur", "pour", "que", "une", "ce",
                           "son", "sa", "ses", "un", "il", "elle", "ils", "elles",
                           "au", "aux", "en", "ne", "pas", "plus", "tout", "comme"}
            found = sum(1 for w in french_words if w in output_lower)
            matched = found >= 3  # At least 3 common French words
            details[dimension] = {"pass": matched, "matched_count": found}
            if not matched:
                all_passed = False
            continue

        # For univers/situation/voix: case-insensitive substring match
        matched_keywords = []
        for kw in keywords:
            if kw.lower() in output_lower:
                matched_keywords.append(kw)

        matched = len(matched_keywords) >= max(1, len(keywords) // 3)  # >= 33% match
        details[dimension] = {
            "pass": matched,
            "matched": matched_keywords,
            "expected": keywords,
        }
        if not matched:
            all_passed = False

    return all_passed, details


def score_output(output: str, prompt_data: dict) -> dict:
    """Score an output using criteria from the prompt data.

    Returns dict with:
      - pass: bool, overall PASS/FAIL
      - details: dict per dimension
      - scores: float scores per dimension (0.0-1.0)
    """
    criteria = prompt_data.get("criteria", {})
    if not criteria:
        # Fallback: no criteria, return neutral
        return {
            "pass": False,
            "details": {},
            "scores": {
                "univers": 0.0,
                "situation": 0.0,
                "voix": 0.0,
                "langue": 0.0,
            },
            "reason": "no_criteria"
        }

    overall, details = extract_criteria_score(output, criteria)

    # Convert to float scores (match ratio per dimension)
    scores = {}
    for dim, d in details.items():
        if dim == "langue":
            scores[dim] = d.get("matched_count", 0) / 3.0
        else:
            matched = len(d.get("matched", []))
            expected = len(d.get("expected", []))
            scores[dim] = matched / max(expected, 1)

    return {
        "pass": overall,
        "details": details,
        "scores": scores,
        "reason": "ok",
    }


def compute_category_scores(results: list[dict]) -> dict[str, Any]:
    """Aggregate scores across a category.

    Returns dict with counts, pass rates, and mean scores per dimension.
    """
    if not results:
        return {}

    count = len(results)
    passed = sum(1 for r in results if r["score"]["pass"])
    pass_rate = passed / count

    # Average scores per dimension
    dims = ["univers", "situation", "voix", "langue"]
    avg_scores = {}
    for dim in dims:
        vals = [r["score"]["scores"].get(dim, 0.0) for r in results]
        avg_scores[dim] = sum(vals) / max(len(vals), 1)

    return {
        "count": count,
        "passed": passed,
        "failed": count - passed,
        "pass_rate": pass_rate,
        "avg_scores": avg_scores,
    }


def run_evaluation(
    model, tokenizer, test_data: list[dict], mode: str = "model",
    max_new_tokens: int = 300, temperature: float = 0.7,
) -> list[dict]:
    """Run full evaluation pipeline.

    Returns list of result dicts with prompt, output, and score.
    """
    results = []

    for i, data in enumerate(test_data):
        prompt_text = data["prompt"]
        print(f"  [{i+1}/{len(test_data)}] Evaluating: {data.get('category', '?')} "
              f"({data.get('univers', '?')} × {data.get('situation', '?')} × {data.get('voice', '?')})",
              end="\r")

        if mode == "baseline":
            output = generate_answer(model, tokenizer, prompt_text,
                                     max_new_tokens, temperature)
        else:
            output = generate_answer(model, tokenizer, prompt_text,
                                     max_new_tokens, temperature)

        score = score_output(output, data)
        results.append({
            "id": data["id"],
            "prompt": prompt_text,
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

    print()  # newline after progress
    return results


def print_results(results: list[dict], label: str = ""):
    """Print formatted evaluation results."""
    print(f"\n{'=' * 70}")
    if label:
        print(f"📊 {label}")
    print('=' * 70)

    # Overall
    passed = sum(1 for r in results if r["score"]["pass"])
    total = len(results)
    print(f"\n  Overall: {passed}/{total} passed ({passed/total*100:.0f}%)")

    # By category
    by_cat = defaultdict(list)
    for r in results:
        by_cat[r["metadata"]["category"]].append(r)

    print(f"\n  {'Category':<25} {'Pass':>6} {'Rate':>8} {'Avg Score':>12}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 8} {'─' * 12}")
    for cat in sorted(by_cat.keys()):
        items = by_cat[cat]
        cat_passed = sum(1 for r in items if r["score"]["pass"])
        cat_rate = cat_passed / len(items)
        avg = sum(r["score"]["scores"].get("univers", 0) +
                  r["score"]["scores"].get("situation", 0) +
                  r["score"]["scores"].get("voix", 0) +
                  r["score"]["scores"].get("langue", 0) for r in items) / len(items)
        avg /= 4
        print(f"  {cat:<25} {cat_passed:>6} {cat_rate:>7.0%} {avg:>11.3f}")


def run_compare(results_baseline: list[dict], results_fine_tuned: list[dict]):
    """Compare baseline vs fine-tuned results and print diff/progression."""
    print(f"\n{'=' * 70}")
    print("📊 TDD COMPARISON: Baseline vs Fine-Tuned")
    print('=' * 70)

    # Overall
    b_passed = sum(1 for r in results_baseline if r["score"]["pass"])
    ft_passed = sum(1 for r in results_fine_tuned if r["score"]["pass"])
    b_total = len(results_baseline)

    print(f"\n  Baseline:   {b_passed}/{b_total} passed ({b_passed/b_total*100:.0f}%)")
    print(f"  Fine-tuned: {ft_passed}/{b_total} passed ({ft_passed/b_total*100:.0f}%)")
    delta = ft_passed - b_passed
    if delta > 0:
        print(f"\n  📈 Progression: +{delta} tests passés ({delta/b_total*100:.0f}pp)")
    elif delta < 0:
        print(f"\n  📉 Regression: {delta} tests perdus")
    else:
        print(f"\n  ➡️  Aucun changement")

    # Per-category comparison
    print(f"\n  {'Category':<25} {'Base':>6} {'FT':>6} {'Diff':>6}")
    print(f"  {'─' * 25} {'─' * 6} {'─' * 6} {'─' * 6}")

    b_by_cat = defaultdict(list)
    ft_by_cat = defaultdict(list)
    for r in results_baseline:
        b_by_cat[r["metadata"]["category"]].append(r)
    for r in results_fine_tuned:
        ft_by_cat[r["metadata"]["category"]].append(r)

    for cat in sorted(set(b_by_cat.keys()) | set(ft_by_cat.keys())):
        b_count = sum(1 for r in b_by_cat.get(cat, []) if r["score"]["pass"])
        ft_count = sum(1 for r in ft_by_cat.get(cat, []) if r["score"]["pass"])
        diff = ft_count - b_count
        marker = "📈" if diff > 0 else ("📉" if diff < 0 else "➡️")
        print(f"  {cat:<25} {b_count:>6} {ft_count:>6} {diff:>+6} {marker}")

    # Detailed diff for each prompt
    print(f"\n{'─' * 70}")
    print("  🔍 DETAILED PROMPT-LEVEL DIFF")
    print(f"{'─' * 70}")

    ft_by_id = {r["id"]: r for r in results_fine_tuned}
    for r_b in results_baseline:
        fid = r_b["id"]
        r_ft = ft_by_id.get(fid)
        if not r_ft:
            continue

        b_status = "✅ PASS" if r_b["score"]["pass"] else "❌ FAIL"
        ft_status = "✅ PASS" if r_ft["score"]["pass"] else "❌ FAIL"

        # Only show prompts where there's a change
        if r_b["score"]["pass"] == r_ft["score"]["pass"]:
            continue

        direction = "✅→❌ REGRESSION" if not r_ft["score"]["pass"] else "❌→✅ FIX"
        print(f"\n  {r_b['metadata']['univers']} × {r_b['metadata']['situation']} × {r_b['metadata']['voice']}")
        print(f"    Prompt: {r_b['prompt'][:70]}...")
        print(f"    Baseline: {b_status}  |  Fine-tuned: {ft_status}")
        print(f"    Direction: {direction}")

        # Show score diff per dimension
        print(f"    Scores:")
        for dim in ["univers", "situation", "voix", "langue"]:
            b_score = r_b["score"]["scores"].get(dim, 0.0)
            ft_score = r_ft["score"]["scores"].get(dim, 0.0)
            diff_s = ft_score - b_score
            if diff_s != 0:
                arrow = "↑" if diff_s > 0 else "↓"
                print(f"      {dim:<10}: {b_score:.2f} {arrow} {ft_score:.2f} ({diff_s:+.2f})")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Suddenly AI Hub LoRA models")
    parser.add_argument(
        "--test-data", default="data/test-prompts.jsonl",
        help="Path to test prompts JSONL file"
    )
    parser.add_argument("--stack", action="store_true",
                       help="Use stacked 3-axis mode")
    parser.add_argument("--adapter-1", help="Univers adapter")
    parser.add_argument("--multiplier-1", type=float, default=1.0)
    parser.add_argument("--adapter-2", help="Situation adapter")
    parser.add_argument("--multiplier-2", type=float, default=1.0)
    parser.add_argument("--adapter-3", help="Voix adapter")
    parser.add_argument("--multiplier-3", type=float, default=1.0)
    parser.add_argument("--full", action="store_true",
                       help="Run full combination sweep (sample output)")
    parser.add_argument("--base", default="Qwen/Qwen2.5-7B-Instruct",
                       help="Base model")
    parser.add_argument("--max-new-tokens", type=int, default=300)
    parser.add_argument("--temperature", type=float, default=0.7)

    # TDD evaluation flags
    parser.add_argument("--baseline", action="store_true",
                       help="Evaluate base model without LoRA (TDD: should FAIL)")
    parser.add_argument("--compare", action="store_true",
                       help="Compare baseline vs fine-tuned (use with --stack)")

    args = parser.parse_args()

    print("🎭 Suddenly AI Hub — Evaluation Pipeline (TDD Strategy)")
    print("=" * 70)

    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test prompts with criteria")

    if args.full:
        print("\nRunning full combination sweep...")
        print("Not implemented - use --compare for TDD evaluation")
        return

    # Initialize for potential comparison
    baseline_results = None
    fine_tuned_results = None

    if args.baseline:
        # === TDD STEP 1: Baseline evaluation (should FAIL) ===
        print(f"\n📋 BASELINE (modèle brut sans LoRA)")
        print(f"   → Expected: FAIL (baseline should not meet RP criteria)")
        print("-" * 70)

        model, tokenizer = load_base_model(args.base)
        baseline_results = run_evaluation(
            model, tokenizer, test_data, mode="baseline",
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        print_results(baseline_results, "Baseline Evaluation (should FAIL)")

        if not args.compare:
            print("\n💡 Run with --compare to also evaluate fine-tuned and see the diff")
            return

    if args.stack:
        if not args.adapter_1 or not args.adapter_2 or not args.adapter_3:
            parser.error("Provide --adapter-1, --adapter-2, --adapter-3 for stacked mode")

        print(f"\n📋 STACKED MODE:")
        print(f"   Univers: {args.adapter_1} × {args.multiplier_1}")
        print(f"   Situation: {args.adapter_2} × {args.multiplier_2}")
        print(f"   Voix: {args.adapter_3} × {args.multiplier_3}")
        print("-" * 70)

        model, tokenizer = load_model_stacked(
            args.adapter_1, args.adapter_2, args.adapter_3,
            args.multiplier_1, args.multiplier_2, args.multiplier_3,
        )

        print(f"\nGenerating responses for fine-tuned model...")
        fine_tuned_results = run_evaluation(
            model, tokenizer, test_data, mode="stacked",
            max_new_tokens=args.max_new_tokens, temperature=args.temperature,
        )
        print_results(fine_tuned_results, "Fine-Tuned Evaluation (should PASS)")

    if args.compare:
        if baseline_results is None:
            # Auto-run baseline if not already done
            print(f"\n📋 Running baseline for comparison...")
            model_b, tokenizer_b = load_base_model(args.base)
            baseline_results = run_evaluation(
                model_b, tokenizer_b, test_data, mode="baseline",
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )

        if fine_tuned_results is None:
            print(f"\n📋 Evaluating fine-tuned model for comparison...")
            if not args.stack:
                parser.error("--compare requires --stack")
            model, tokenizer = load_model_stacked(
                args.adapter_1, args.adapter_2, args.adapter_3,
                args.multiplier_1, args.multiplier_2, args.multiplier_3,
            )
            fine_tuned_results = run_evaluation(
                model, tokenizer, test_data, mode="stacked",
                max_new_tokens=args.max_new_tokens, temperature=args.temperature,
            )
            print_results(fine_tuned_results, "Fine-Tuned Evaluation (should PASS)")

        if baseline_results and fine_tuned_results:
            run_compare(baseline_results, fine_tuned_results)


if __name__ == "__main__":
    main()
