#!/usr/bin/env python3
"""Evaluation pipeline for Suddenly AI Hub stacked LoRA models.

Usage:
  python scripts/evaluate.py --test-data data/test-prompts.jsonl
  python scripts/evaluate.py --test-data data/test-prompts.jsonl --stack \
    --adapter-1 fantasy-medievale --multiplier-1 1.0 \
    --adapter-2 combat            --multiplier-2 1.0 \
    --adapter-3 narquois          --multiplier-3 1.0
  python scripts/evaluate.py --full  # Run all combinations

Each prompt is scored on:
  - relevance: Does the output match the prompt intent?
  - style: Does the output match the requested style/voice?
  - immersion: Does the output feel like authentic RP?
  - creativity: Is the output interesting and non-generic?
"""
import argparse
import json
import sys
from pathlib import Path
from typing import Any

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "scripts"))


def load_test_data(path: str) -> list[dict]:
    """Load JSONL test prompts."""
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


def generate_baseline(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    """Generate from base model without any LoRA."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto"
    )
    return generate_answer(model, tokenizer, prompt, max_new_tokens, temperature)


def score_output(output: str, prompt_data: dict, category: str) -> dict[str, float]:
    """Score an output against the expected dimensions.

    In production, this would use LLM-as-judge or human raters.
    For now, return placeholder scores that can be replaced.

    Scores are 0.0–1.0 for each dimension.
    """
    # Placeholder: real implementation needs LLM-as-judge or human evaluation
    # This is a template for the scoring framework
    return {
        "relevance": 0.0,    # Matches prompt intent
        "style": 0.0,        # Matches requested style/voice
        "immersion": 0.0,    # Authentic RP quality
        "creativity": 0.0,   # Non-generic, interesting
        "length_score": 0.0, # Adequate length (30–300 chars ideal)
    }


def compute_category_scores(scores: list[dict]) -> dict[str, float]:
    """Aggregate scores across a category."""
    import numpy as np
    if not scores:
        return {}
    result = {}
    keys = scores[0].keys()
    for key in keys:
        vals = [s[key] for s in scores]
        result[f"{key}_mean"] = np.mean(vals)
        result[f"{key}_std"] = np.std(vals)
        result[f"{key}_min"] = np.min(vals)
        result[f"{key}_max"] = np.max(vals)
    return result


def run_evaluation(
    model, tokenizer, test_data: list[dict], max_new_tokens: int = 300,
    temperature: float = 0.7, mode: str = "stacked",
) -> dict:
    """Run full evaluation pipeline.

    Returns results dict with per-prompt and per-category scores.
    """
    results = []
    prompts = list(test_data)  # Make a copy

    for i, data in enumerate(prompts):
        prompt_text = data["prompt"]
        data["output"] = generate_answer(
            model, tokenizer, prompt_text, max_new_tokens, temperature
        )
        data["scores"] = score_output(data["output"], data, data.get("category", "unknown"))
        results.append(data)

        # Progress
        if (i + 1) % 10 == 0:
            print(f"  Evaluated {i + 1}/{len(prompts)} prompts...")

    return results


def print_results(results: list[dict]):
    """Print formatted evaluation results."""
    # Per-category aggregation
    categories: dict[str, list[dict]] = {}
    for r in results:
        cat = r.get("category", "unknown")
        categories.setdefault(cat, []).append(r)

    print("\n" + "=" * 70)
    print("📊 Evaluation Results by Category")
    print("=" * 70)

    for cat in sorted(categories.keys()):
        items = categories[cat]
        cat_scores = compute_category_scores(
            [item["scores"] for item in items]
        )
        print(f"\n  {cat}:")
        print(f"    Count: {len(items)}")
        for key in ["relevance", "style", "immersion", "creativity"]:
            mean_key = f"{key}_mean"
            std_key = f"{key}_std"
            if mean_key in cat_scores:
                print(f"    {key:12s}: {cat_scores[mean_key]:.3f} ± {cat_scores[std_key]:.3f}")


def run_full_combinations(test_data: list[dict]):
    """Evaluate all 100 adapter combinations."""
    adapters = {
        "univers": [
            ("fantasy-medievale", 1.0),
            ("cyberpunk", 1.0),
        ],
        "situation": [
            ("combat", 1.0),
            ("romance", 1.0),
        ],
        "voice": [
            ("solennel", 1.0),
            ("narquois", 1.0),
        ],
    }

    combos = []
    for u in adapters["univers"]:
        for s in adapters["situation"]:
            for v in adapters["voice"]:
                combos.append((u, s, v))

    print(f"Evaluating {len(combos)} combinations on {len(test_data)} prompts...")
    print(f"Sample output for first 2 combos:")

    for i, (u, s, v) in enumerate(combos[:2]):
        print(f"\n  Combo {i + 1}: {u[0]} × {s[0]} × {v[0]}")
        try:
            model, tokenizer = load_model_stacked(
                f"suddenly-{u[0]}", f"suddenly-{s[0]}", f"suddenly-{v[0]}",
                u[1], s[1], v[1],
            )
            sample = generate_answer(model, tokenizer, test_data[0]["prompt"])
            print(f"    Prompt: {test_data[0]['prompt'][:60]}...")
            print(f"    Output: {sample[:100]}...")
        except Exception as e:
            print(f"    Error: {e}")


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
    args = parser.parse_args()

    print("🎭 Suddenly AI Hub — Evaluation Pipeline")
    print("=" * 70)

    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test prompts")

    if args.full:
        print("\nRunning full combination sweep...")
        run_full_combinations(test_data)
        return

    if args.stack:
        print(f"\nStacked mode:")
        print(f"  Univers: {args.adapter_1} × {args.multiplier_1}")
        print(f"  Situation: {args.adapter_2} × {args.multiplier_2}")
        print(f"  Voix: {args.adapter_3} × {args.multiplier_3}")

        model, tokenizer = load_model_stacked(
            args.adapter_1, args.adapter_2, args.adapter_3,
            args.multiplier_1, args.multiplier_2, args.multiplier_3,
        )
        mode = "stacked"
    else:
        parser.error("Provide --stack for stacked evaluation or --full for combination sweep")

    print(f"\nGenerating responses...")
    results = run_evaluation(
        model, tokenizer, test_data,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        mode=mode,
    )

    print_results(results)


if __name__ == "__main__":
    main()
