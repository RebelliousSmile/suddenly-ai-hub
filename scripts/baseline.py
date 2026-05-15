#!/usr/bin/env python3
"""Baseline evaluation — generates responses from the base model WITHOUT any LoRA adapter.

Usage:
  python scripts/baseline.py --test-data data/test-prompts.jsonl
  python scripts/baseline.py --test-data data/test-prompts.jsonl --max-new-tokens 500
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np


def load_test_data(path: str) -> list[dict]:
    """Load JSONL test prompts."""
    prompts = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                prompts.append(json.loads(line))
    return prompts


def generate_baseline(prompt: str, max_new_tokens: int = 300, temperature: float = 0.7) -> str:
    """Generate from base model without any LoRA adapter."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("  Loading base model Qwen/Qwen2.5-7B-Instruct (no LoRA)...")
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto"
    )

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


def score_output(output: str, prompt_data: dict) -> dict[str, float]:
    """Score an output — placeholder for LLM-as-judge or human evaluation."""
    return {
        "relevance": 0.0,
        "style": 0.0,
        "immersion": 0.0,
        "creativity": 0.0,
        "length_score": 0.0,
    }


def compute_category_scores(scores: list[dict]) -> dict[str, float]:
    """Aggregate scores across a category."""
    if not scores:
        return {}
    result = {}
    keys = scores[0].keys()
    for key in keys:
        vals = [s[key] for s in scores]
        result[f"{key}_mean"] = float(np.mean(vals))
        result[f"{key}_std"] = float(np.std(vals))
        result[f"{key}_min"] = float(np.min(vals))
        result[f"{key}_max"] = float(np.max(vals))
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Baseline evaluation: base model without LoRA"
    )
    parser.add_argument(
        "--test-data", default="data/test-prompts.jsonl",
        help="Path to test prompts JSONL file"
    )
    parser.add_argument("--max-new-tokens", type=int, default=300,
                        help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--output", default="baseline-results.json",
                        help="Output JSON file for results")
    args = parser.parse_args()

    print("🎭 Suddenly AI Hub — Baseline (No LoRA)")
    print("=" * 70)

    test_data = load_test_data(args.test_data)
    print(f"Loaded {len(test_data)} test prompts")

    results = []
    for i, data in enumerate(test_data):
        print(f"\n  [{i + 1}/{len(test_data)}] Category: {data.get('category', 'unknown')}")
        print(f"    Prompt: {data['prompt'][:80]}...")

        start = __import__("time").time()
        data["baseline_output"] = generate_baseline(
            data["prompt"], args.max_new_tokens, args.temperature
        )
        elapsed = __import__("time").time() - start
        data["baseline_time"] = elapsed

        print(f"    Output: {data['baseline_output'][:120]}...")
        print(f"    Time: {elapsed:.1f}s")

        data["scores"] = score_output(data["baseline_output"], data)
        results.append(data)

    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
