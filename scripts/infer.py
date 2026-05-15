#!/usr/bin/env python3
"""Local inference script for Suddenly AI Hub LoRA adapters."""
import argparse
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

# Default base model
DEFAULT_BASE = "Qwen/Qwen2.5-7B-Instruct"


def load_model(adapter_path: str, base_model: str = DEFAULT_BASE):
    """Load base model + LoRA adapter."""
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
    parser = argparse.ArgumentParser(description="Inference with Suddenly AI Hub LoRA")
    parser.add_argument("--adapter", required=True, help="Path to LoRA adapter directory")
    parser.add_argument("--prompt", required=True, help="Input prompt")
    parser.add_argument("--base", default=DEFAULT_BASE, help="Base model (default: Qwen2.5-7B-Instruct)")
    parser.add_argument("--max-new-tokens", type=int, default=300, help="Max new tokens")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--role", choices=["user", "assistant", "system"], default="user", help="Role for prompt")
    args = parser.parse_args()

    model, tokenizer = load_model(args.adapter, args.base)

    # Format prompt with role
    if args.role == "system":
        full_prompt = f"<|im_start|>system\n{args.prompt}<|im_end|>\n<|im_start|>user\n"
    else:
        full_prompt = f"<|im_start|>user\n{args.prompt}<|im_end|>\n<|im_start|>assistant\n"

    print(f"\n{'='*60}")
    print(f"Adapter: {args.adapter}")
    print(f"{'='*60}")
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
    if "<|im_start|>assistant\n" in result:
        response = result.split("<|im_start|>assistant\n")[-1]
    else:
        response = result

    print(response.strip())
    print()


if __name__ == "__main__":
    main()
