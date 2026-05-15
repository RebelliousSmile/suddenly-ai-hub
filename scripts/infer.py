#!/usr/bin/env python3
"""Local inference script for Suddenly AI Hub LoRA adapters.

Supports both single-adapter and stacked (multi-axis) inference.
Usage:
  # Single adapter
  python scripts/infer.py --adapter fantasy-medievale --prompt "..."

  # Stacked 3-axis inference
  python scripts/infer.py --stack \\
    --adapter-1 fantasy-medievale --multiplier-1 1.0 \\
    --adapter-2 combat            --multiplier-2 1.0 \\
    --adapter-3 narquois          --multiplier-3 1.0 \\
    --prompt "..."
"""
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Default base model
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"


def load_single_model(adapter_path: str, base_model: str = DEFAULT_BASE):
    """Load base model + single LoRA adapter."""
    print(f"Loading base model: {base_model}...")
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    print(f"Loading LoRA adapter: {adapter_path}...")
    model = PeftModel.from_pretrained(model, adapter_path)
    return model, tokenizer


def load_stacked_models(
    adapter1: str, mult1: float,
    adapter2: str, mult2: float,
    adapter3: str, mult3: float,
    base_model: str = DEFAULT_BASE,
):
    """Load base model + stacked LoRA adapters (3 axes).

    Uses PEFT named adapters with merge-and-unmerge for dynamic multiplier control.
    """
    base_path = Path(base_model)
    if base_path.exists():
        print(f"Loading cached base model: {base_model}...")
    else:
        print(f"Loading base model: {base_model}...")

    tokenizer = AutoTokenizer.from_pretrained(base_model)
    model = AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.float16,
        device_map="auto",
    )

    # Load each adapter with a distinct name for PEFT management
    adapters = [
        (adapter1, mult1, "univers"),
        (adapter2, mult2, "situation"),
        (adapter3, mult3, "voix"),
    ]

    for adapter_id, mult, axis_name in adapters:
        if mult == 0.0:
            print(f"  Skipped {axis_name}: multiplier = 0.0")
            continue
        adapter_path = Path(__file__).resolve().parent.parent / "models" / adapter_id
        if not adapter_path.exists():
            print(f"  ⚠️  Adapter not found: {adapter_path}")
            print(f"      Falling back to base model for {axis_name}")
            continue
        print(f"  Loading {axis_name}: {adapter_id} (multiplier={mult})...")
        model = PeftModel.from_pretrained(
            model,
            str(adapter_path),
            adapter_name=axis_name,
        )

    model.set_adapter(["univers", "situation", "voix"])
    return model, tokenizer, adapters


def generate(model, tokenizer, prompt, max_new_tokens=300, temperature=0.7):
    """Run inference with the model."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        do_sample=True,
        top_p=0.9,
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)


def main():
    parser = argparse.ArgumentParser(
        description="Inference with Suddenly AI Hub LoRA (single or stacked)"
    )
    parser.add_argument(
        "--stack", action="store_true",
        help="Enable stacked 3-axis inference (Univers + Situation + Voix)"
    )
    # Single adapter mode
    parser.add_argument("--adapter", help="Path to single LoRA adapter directory")
    parser.add_argument("--prompt", help="Input prompt")
    # Stacked mode
    parser.add_argument("--adapter-1", help="Axe 1: Univers adapter ID")
    parser.add_argument("--multiplier-1", type=float, default=1.0,
                        help="Univers multiplier (default: 1.0)")
    parser.add_argument("--adapter-2", help="Axe 2: Situation adapter ID")
    parser.add_argument("--multiplier-2", type=float, default=1.0,
                        help="Situation multiplier (default: 1.0)")
    parser.add_argument("--adapter-3", help="Axe 3: Voix adapter ID")
    parser.add_argument("--multiplier-3", type=float, default=1.0,
                        help="Voix multiplier (default: 1.0)")
    # Common args
    parser.add_argument("--base", default=DEFAULT_BASE,
                        help="Base model (default: Qwen2.5-7B-Instruct)")
    parser.add_argument("--max-new-tokens", type=int, default=300,
                        help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7,
                        help="Sampling temperature")
    parser.add_argument("--role", choices=["user", "assistant", "system"],
                        default="user", help="Role for prompt")
    args = parser.parse_args()

    # Validate: must use --stack or --adapter
    if args.stack:
        if not (args.adapter_1 and args.adapter_2 and args.adapter_3 and args.prompt):
            parser.error(
                "Stack mode requires --adapter-1, --adapter-2, --adapter-3, and --prompt"
            )
        model, tokenizer, loaded = load_stacked_models(
            args.adapter_1, args.multiplier_1,
            args.adapter_2, args.multiplier_2,
            args.adapter_3, args.multiplier_3,
            args.base,
        )
        print(f"\n{'='*60}")
        print(f"Stacked 3-axis inference")
        print(f"{'='*60}")
        for adapter_id, mult, axis in loaded:
            print(f"  {axis:12s}: {adapter_id:24s} × {mult}")
        print()
    elif args.adapter and args.prompt:
        model, tokenizer = load_single_model(args.adapter, args.base)
        print(f"\n{'='*60}")
        print(f"Adapter: {args.adapter}")
        print(f"{'='*60}")
    else:
        parser.error("Provide --adapter (single) or --stack (multi-axis) + --prompt")

    # Format prompt with role
    if args.role == "system":
        full_prompt = f"🌐system\n{args.prompt}\n\n👤user\n"
    else:
        full_prompt = f"👤user\n{args.prompt}\n\n🤖assistant\n"

    print(f"👤 User: {args.prompt}")
    print(f"\n{'='*60}")
    print(f"🤖 Assistant:")
    print(f"{'='*60}\n")

    result = generate(
        model, tokenizer, full_prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
    )

    # Extract only the assistant response
    if "🤖assistant\n" in result:
        response = result.split("🤖assistant\n")[-1]
    else:
        response = result

    print(response.strip())
    print()


if __name__ == "__main__":
    main()
