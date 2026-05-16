#!/usr/bin/env python3
"""
Diff Runner for Suddenly AI Hub LoRA adapters (Ollama-based).

Compares Base Model vs Fine-Tuned LoRA responses for each use case.
Uses Ollama local inference — no PyTorch required.

Usage:
    # Run all tests against Ollama
    python scripts/run-diff.py

    # Run specific scenario
    python scripts/run-diff.py --scenario dialogue

    # Use different models
    python scripts/run-diff.py --base-model qwen2.5:7b-instruct-q4_K_M \
                               --ft-model suddenly-7b
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path


DEFAULT_BASE = "qwen2.5:7b-instruct-q4_K_M"
SCENARIOS_FILE = Path(__file__).parent.parent / "data" / "test-scenarios.jsonl"


def run_ollama(prompt: str, model: str, max_tokens: int = 300, temperature: float = 0.7) -> tuple[str, float]:
    """Run inference via Ollama CLI and return (response_text, elapsed_seconds)."""
    start_time = time.time()
    cmd = [
        "ollama", "run", model,
        f"--num-predict", str(max_tokens),
        f"--temperature", str(temperature),
        prompt,
    ]
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        elapsed = time.time() - start_time
        # Ollama output format: "model\n\nresponse" or just "response"
        text = result.stdout.strip()
        if text.startswith(model + "\n\n"):
            text = text[len(model) + 2:]
        return text, elapsed
    except subprocess.TimeoutExpired:
        return "⚠️ Timeout (>120s)", time.time() - start_time
    except FileNotFoundError:
        return "⚠️ Ollama not found. Install with: brew install ollama (mac) or follow ollama.ai", 0
    except Exception as e:
        return f"⚠️ Error: {e}", 0


def format_prompt(scenario: dict) -> str:
    """Build a Suddenly-style prompt from a scenario dict."""
    role = scenario.get("role", "user")
    prompt_text = scenario["prompt"]
    
    # Add context tags
    tags = []
    if scenario.get("adapter"):
        tags.append(f"[Axe Univers: {scenario['adapter']}]")
    if scenario.get("situation"):
        tags.append(f"[Situation: {scenario['situation']}]")
    
    # System instruction — guide the model toward Suddenly RP style
    system = "Tu es un assistant de jeu de rôle RP en français. Réponds dans le style narratif du RP français, avec un ton immersif et des descriptions riches. Utilise le français standard sans anglicismes."
    
    context = " ".join(tags) if tags else ""
    
    if context:
        return f"{system}\n\n{context}\n\n{role}: {prompt_text}\n\nassistant:"
    return f"{system}\n\n{role}: {prompt_text}\n\nassistant:"


def run_diff_scenario(base_prompt: str, ft_prompt: str, scenario: dict, args):
    """Run base and fine-tuned models on a single scenario."""
    scenario_name = scenario["name"]
    adapter = scenario.get("adapter", "base")
    situation = scenario.get("situation", "")
    prompt_text = scenario["prompt"]

    print(f"\n{'='*70}")
    print(f"📝 {scenario_name}")
    print(f"{'='*70}")
    if adapter != "base":
        print(f"   Univers: {adapter} | Situation: {situation}")
    print(f"   Prompt: {prompt_text}")
    print(f"{'─'*70}")

    # Base model inference
    print(f"\n🔵 Base ({args.base_model}):")
    print("─" * 40)
    base_response, base_time = run_ollama(
        base_prompt, args.base_model, args.max_tokens, args.temperature
    )
    print(base_response)
    print(f"\n⏱ {base_time:.1f}s")

    # Fine-tuned model inference
    print(f"\n🟣 Fine-Tuned ({args.ft_model} — {adapter}):")
    print("─" * 40)
    ft_response, ft_time = run_ollama(
        ft_prompt, args.ft_model, args.max_tokens, args.temperature
    )
    print(ft_response)
    print(f"\n⏱ {ft_time:.1f}s")

    # Summary
    speed_diff = ((base_time - ft_time) / base_time * 100) if base_time > 0 else 0
    print(f"\n{'─'*70}")
    print(f"📊 Diff: {speed_diff:+.1f}% vitesse | Tokens estimés: base~{len(base_response)//4} | ft~{len(ft_response)//4}")
    print(f"{'='*70}")

    # Export if requested
    if args.export:
        export_entry = {
            "scenario": scenario_name,
            "adapter": adapter,
            "situation": situation,
            "prompt": prompt_text,
            "base_response": base_response,
            "ft_response": ft_response,
            "base_time": base_time,
            "ft_time": ft_time,
        }
        args.export.write(json.dumps(export_entry, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Compare Base vs Fine-Tuned LoRA responses via Ollama"
    )
    parser.add_argument(
        "--base-model", default=DEFAULT_BASE,
        help=f"Base Ollama model (default: {DEFAULT_BASE})"
    )
    parser.add_argument(
        "--ft-model", default="suddenly-7b",
        help="Fine-tuned Ollama model name (default: suddenly-7b)"
    )
    parser.add_argument(
        "--scenario", default=None,
        help="Run only this specific scenario by ID"
    )
    parser.add_argument(
        "--max-tokens", type=int, default=300,
        help="Max output tokens (default: 300)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Sampling temperature (default: 0.7)"
    )
    parser.add_argument(
        "--export", type=argparse.FileType("a", encoding="utf-8"), default=None,
        metavar="FILE",
        help="Append results to JSONL file"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show prompts only, don't run inference"
    )
    args = parser.parse_args()

    # Load scenarios
    scenarios_path = SCENARIOS_FILE
    if not scenarios_path.exists():
        print(f"Error: Scenarios file not found: {scenarios_path}")
        sys.exit(1)

    scenarios = []
    with open(scenarios_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                scenarios.append(json.loads(line))

    # Filter if specific scenario requested
    if args.scenario:
        scenarios = [s for s in scenarios if s["scenario_id"] == args.scenario]
        if not scenarios:
            print(f"Error: No scenario found with ID '{args.scenario}'")
            print(f"Available: {[s['scenario_id'] for s in scenarios]}")
            sys.exit(1)

    print(f"📦 Loaded {len(scenarios)} scenarios from {scenarios_path}")
    if args.scenario:
        print(f"🎯 Filtered to: {args.scenario}")
    print(f"🔵 Base model: {args.base_model}")
    print(f"🟣 Fine-tuned model: {args.ft_model}")
    print(f"🌡 Temperature: {args.temperature} | Max tokens: {args.max_tokens}")

    # Check Ollama availability
    try:
        subprocess.run(["ollama", "list"], capture_output=True, check=True, timeout=5)
    except (FileNotFoundError, subprocess.TimeoutExpired, subprocess.CalledProcessError):
        print("\n⚠️  Ollama seems unavailable.")
        print("   Make sure 'ollama' is in your PATH and at least one model is pulled.")
        print("   Pull base model with: ollama pull qwen2.5:7b-instruct-q4_K_M")
        print("   Pull FT model with: ollama pull suddenly-7b")
        sys.exit(1)

    for scenario in scenarios:
        base_prompt = format_prompt({"role": "user", "prompt": scenario["prompt"]})
        ft_prompt = format_prompt(scenario)

        if args.dry_run:
            print(f"\n{'='*70}")
            print(f"📝 DRY RUN: {scenario['name']}")
            print(f"{'='*70}")
            print(f"\nBase prompt:\n{base_prompt}")
            print(f"\nFine-tuned prompt:\n{ft_prompt}")
            print(f"{'='*70}")
            continue

        run_diff_scenario(base_prompt, ft_prompt, scenario, args)

    print("\n✅ All scenarios completed!")


if __name__ == "__main__":
    main()
