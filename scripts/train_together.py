#!/usr/bin/env python3
"""
train_together.py — Unified Together.ai fine-tuning client.

Subcommands:
  validate   Validate JSONL input + split train/val
  train      Upload + start fine-tuning job
  infer      Run inference on val examples with fine-tuned model

Usage:
  python scripts/train_together.py validate --input data/test-dataset-rp.jsonl
  python scripts/train_together.py train [--model <model_id>] [--lr <lr>] [--epochs <n>]
  python scripts/train_together.py infer [--model-id <override>]

Model list (fine-tunable on Together.ai):
  - Qwen/Qwen2.5-7B-Instruct
  - meta-llama/Llama-3.1-70B-Instruct-Turbo
  - Qwen/Qwen2.5-14B-Instruct
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path
from typing import Any, Optional

import requests

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = (
    "Tu es un conteur de roleplay. Réponds en français, "
    "dans un registre narratif et immersif."
)

FINE_TUNABLE_MODELS = [
    "Qwen/Qwen2.5-7B-Instruct",
    "meta-llama/Llama-3.1-70B-Instruct-Turbo",
    "Qwen/Qwen2.5-14B-Instruct",
]

CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
CACHE_MODEL_ID_FILE = CACHE_DIR / "model_id.txt"

DEFAULT_MODEL = FINE_TUNABLE_MODELS[0]
DEFAULT_LR = 2.0
DEFAULT_EPOCHS = 3
DEFAULT_TIMEOUT = 3600  # 1 hour
POLL_INTERVAL = 30  # seconds
DATA_DIR = Path(__file__).resolve().parent.parent / "data"

TOGETHER_API_BASE = "https://api.together.xyz/v1"


def _api_headers() -> dict[str, str]:
    """Return headers with Together.ai API key from environment."""
    api_key = os.environ.get("TOGETHER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    if not api_key:
        print("❌ TOGETHER_API_KEY not set in environment.", file=sys.stderr)
        sys.exit(1)
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }


def _api_get(path: str) -> dict:
    resp = requests.get(f"{TOGETHER_API_BASE}{path}", headers=_api_headers())
    resp.raise_for_status()
    return resp.json()


def _api_post(path: str, json_data: Optional[dict] = None) -> dict:
    resp = requests.post(
        f"{TOGETHER_API_BASE}{path}",
        headers=_api_headers(),
        json=json_data,
    )
    resp.raise_for_status()
    return resp.json()


def _api_upload_file(file_path: Path) -> str:
    """Upload a file for fine-tuning. Returns file_id."""
    api_key = os.environ.get("TOGETHER_API_KEY", os.environ.get("OPENAI_API_KEY", ""))
    url = f"{TOGETHER_API_BASE}/files"
    with open(file_path, "rb") as f:
        resp = requests.post(
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            files={"file": (file_path.name, f, "application/json")},
            data={"purpose": "fine-tune"},
        )
    resp.raise_for_status()
    result = resp.json()
    return result["id"]


def _ensure_cache_dir() -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)


def _persist_model_id(model_id: str) -> None:
    _ensure_cache_dir()
    CACHE_MODEL_ID_FILE.write_text(model_id, encoding="utf-8")


def _load_model_id() -> Optional[str]:
    if CACHE_MODEL_ID_FILE.exists():
        return CACHE_MODEL_ID_FILE.read_text(encoding="utf-8").strip()
    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def _validate_message(msg: dict) -> bool:
    """Check a single message dict has role and content."""
    if not isinstance(msg, dict):
        return False
    if "role" not in msg or "content" not in msg:
        return False
    if not isinstance(msg["role"], str) or not msg["role"]:
        return False
    if not isinstance(msg["content"], str) or not msg["content"]:
        return False
    return True


def _validate_example(example: dict) -> tuple[bool, str]:
    """Validate a single JSONL example. Returns (ok, error_message)."""
    if not isinstance(example, dict):
        return False, "Not a JSON object"
    if "messages" not in example:
        return False, "Missing 'messages' field"
    messages = example["messages"]
    if not isinstance(messages, list) or len(messages) == 0:
        return False, "'messages' must be a non-empty array"
    for i, msg in enumerate(messages):
        if not _validate_message(msg):
            return False, f"Invalid message at index {i}: {json.dumps(msg)[:100]}"
    return True, ""


# ---------------------------------------------------------------------------
# Subcommand: validate
# ---------------------------------------------------------------------------


def cmd_validate(args: argparse.Namespace) -> None:
    """Validate JSONL input, inject system prompt if missing, split train/val."""
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Input file not found: {input_path}", file=sys.stderr)
        sys.exit(1)

    system_prompt = args.system if args.system else DEFAULT_SYSTEM_PROMPT
    examples = []
    line_num = 0
    errors = []

    with open(input_path, "r", encoding="utf-8") as f:
        for line in f:
            line_num += 1
            line = line.strip()
            if not line:
                continue
            try:
                example = json.loads(line)
            except json.JSONDecodeError as e:
                errors.append(f"Line {line_num}: invalid JSON — {e}")
                continue
            ok, err = _validate_example(example)
            if not ok:
                errors.append(f"Line {line_num}: {err}")
                continue

            # Inject system prompt if none exists
            has_system = any(m.get("role") == "system" for m in example["messages"])
            if not has_system:
                example["messages"].insert(0, {
                    "role": "system",
                    "content": system_prompt,
                })
            examples.append(example)

    # Report errors
    if errors:
        print(f"⚠️  {len(errors)} validation error(s):", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)

    if not examples:
        print("❌ No valid examples found.", file=sys.stderr)
        sys.exit(1)

    n = len(examples)
    random.seed(args.seed)
    indices = list(range(n))
    random.shuffle(indices)
    split = int(n * args.split_ratio)
    train_idx = set(indices[:split])

    train_data = [examples[i] for i in sorted(train_idx)]
    val_data = [examples[i] for i in sorted(indices[split:])]

    print(f"✅ Validation complete: {n} valid examples")
    print(f"   Train: {len(train_data)} examples")
    print(f"   Val:   {len(val_data)} examples")

    if args.dry_run:
        print("   (dry run — no files written)")
        return

    # Write train
    train_out = DATA_DIR / "train.jsonl"
    with open(train_out, "w", encoding="utf-8") as f:
        for ex in train_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"   → {train_out}")

    # Write val
    val_out = DATA_DIR / "val.jsonl"
    with open(val_out, "w", encoding="utf-8") as f:
        for ex in val_data:
            f.write(json.dumps(ex, ensure_ascii=False) + "\n")
    print(f"   → {val_out}")


# ---------------------------------------------------------------------------
# Subcommand: train
# ---------------------------------------------------------------------------


def _list_fine_tunable_models() -> list[str]:
    """Try to fetch available models from Together.ai API. Returns list."""
    try:
        data = _api_get("/models")
        model_ids = [m["id"] for m in data if isinstance(m, dict) and "id" in m]
        # Filter to only models in our known fine-tunable list
        known_set = set(FINE_TUNABLE_MODELS)
        available = [mid for mid in model_ids if mid in known_set]
        # Return all known fine-tunable models, sorted by preference
        result = []
        for m in FINE_TUNABLE_MODELS:
            if m in available or m in model_ids:
                result.append(m)
        return result
    except Exception:
        # If API call fails, return hardcoded list
        return list(FINE_TUNABLE_MODELS)


def cmd_train(args: argparse.Namespace) -> None:
    """Upload dataset and start fine-tuning job on Together.ai."""
    train_file = DATA_DIR / "train.jsonl"
    if not train_file.exists():
        print(f"❌ Training data not found: {train_file}", file=sys.stderr)
        print("   Run 'validate' first.", file=sys.stderr)
        sys.exit(1)

    model = args.model if args.model else DEFAULT_MODEL

    # Check model is fine-tunable
    known_set = set(FINE_TUNABLE_MODELS)
    if model not in known_set:
        # Try to get actual available models
        available = _list_fine_tunable_models()
        available_set = set(available)
        print(f"⚠️  Model '{model}' not in known fine-tunable list.")
        if available:
            print(f"   Available fine-tunable models: {', '.join(available)}")
        else:
            print(f"   Known fine-tunable models: {', '.join(FINE_TUNABLE_MODELS)}")
        # Use first known as fallback
        fallback = FINE_TUNABLE_MODELS[0]
        print(f"   Falling back to '{fallback}'.")
        model = fallback
        if args.model is not None:
            print(f"   (Use --model to override, or --model {fallback} to use default)")

    print(f"🚀 Starting fine-tuning:")
    print(f"   Model: {model}")
    print(f"   Learning rate multiplier: {args.lr}")
    print(f"   Epochs: {args.epochs}")
    print(f"   Timeout: {args.timeout}s")

    # Step 1: Upload file
    print("\n📤 Uploading training file...")
    try:
        file_id = _api_upload_file(train_file)
        print(f"   File uploaded: {file_id}")
    except requests.HTTPError as e:
        print(f"❌ Upload failed: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"   Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)

    # Step 2: Create training job
    print("\n🏋️  Creating training job...")
    try:
        job_data = {
            "model": model,
            "training_files": [file_id],
            "learning_rate_multiplier": args.lr,
            "n_epochs": args.epochs,
        }
        resp = _api_post("/fine_tuning/jobs", json_data=job_data)
        job_id = resp.get("id", resp.get("job_id", "unknown"))
        print(f"   Job created: {job_id}")
        print(f"   Status: {resp.get('status', 'unknown')}")
    except requests.HTTPError as e:
        print(f"❌ Job creation failed: {e}", file=sys.stderr)
        if e.response is not None:
            print(f"   Response: {e.response.text}", file=sys.stderr)
        sys.exit(1)

    # Step 3: Poll for completion
    print(f"\n⏳ Polling job status (timeout: {args.timeout}s, interval: {POLL_INTERVAL}s)...")
    start_time = time.time()
    while True:
        elapsed = time.time() - start_time
        if elapsed > args.timeout:
            print(f"❌ Timeout after {int(elapsed)}s. Job may still be running.", file=sys.stderr)
            sys.exit(1)

        try:
            status_data = _api_get(f"/fine_tuning/jobs/{job_id}")
            status = status_data.get("status", "UNKNOWN")
            trained_weights = status_data.get("trained_weights", {})
            result_model_id = status_data.get("result_model_id")

            if status == "SUCCEEDED":
                print(f"✅ Training complete! (in {int(elapsed)}s)")
                print(f"   Job: {job_id}")
                print(f"   Result model ID: {result_model_id}")
                if result_model_id:
                    _persist_model_id(result_model_id)
                    print(f"   Model ID persisted to {CACHE_MODEL_ID_FILE}")
                print(f"\n🎉 Fine-tuning successful!")
                print(f"\nNext steps:")
                print(f"  python scripts/train_together.py infer")
                print(f"  (reads model ID from {CACHE_MODEL_ID_FILE})")
                return
            elif status == "FAILED":
                error_msg = status_data.get("error", "Unknown error")
                print(f"❌ Training failed: {error_msg}", file=sys.stderr)
                sys.exit(1)
            else:
                # Progress info
                progress = status_data.get("training_triggers", {})
                train_steps = status_data.get("training_steps", 0)
                loss = status_data.get("training_loss", None)
                progress_str = f" {train_steps} steps" if train_steps else ""
                loss_str = f" loss={loss}" if loss else ""
                print(f"   [{int(elapsed)}s] {status}{progress_str}{loss_str}", file=sys.stderr)
        except requests.HTTPError as e:
            print(f"   [{int(elapsed)}s] Poll error: {e}", file=sys.stderr)
            time.sleep(POLL_INTERVAL)

        time.sleep(POLL_INTERVAL)


# ---------------------------------------------------------------------------
# Subcommand: infer
# ---------------------------------------------------------------------------


def cmd_infer(args: argparse.Namespace) -> None:
    """Run inference on val examples using the fine-tuned model."""
    # Get model ID: first try --model-id override, then cache file
    model_id = args.model_id if args.model_id else _load_model_id()
    if not model_id:
        print(f"❌ No model ID found.", file=sys.stderr)
        print(f"   Run 'train' first, or use --model-id <model_id>", file=sys.stderr)
        sys.exit(1)

    val_file = DATA_DIR / "val.jsonl"
    if not val_file.exists():
        print(f"❌ Validation data not found: {val_file}", file=sys.stderr)
        print("   Run 'validate' first.", file=sys.stderr)
        sys.exit(1)

    # Read val examples
    examples = []
    with open(val_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if not examples:
        print("❌ No val examples found.", file=sys.stderr)
        sys.exit(1)

    predictions = []
    print(f"🔮 Running inference on {len(examples)} val examples:")
    print(f"   Model: {model_id}")

    for i, ex in enumerate(examples):
        messages = ex["messages"]
        try:
            resp = _api_post("/chat/completions", json_data={
                "model": model_id,
                "messages": messages,
                "max_tokens": 1024,
                "temperature": 0.7,
            })
            completion = resp["choices"][0]["message"]["content"]
            print(f"   [{i+1}/{len(examples)}] ✅")
        except requests.HTTPError as e:
            completion = f"[ERROR] {e}"
            print(f"   [{i+1}/{len(examples)}] ❌ {e}")
            if e.response is not None:
                print(f"      {e.response.text}", file=sys.stderr)

        predictions.append({
            "messages": messages,
            "prediction": completion,
            "model": model_id,
        })

    # Write predictions
    pred_file = DATA_DIR / "val-predictions.jsonl"
    with open(pred_file, "w", encoding="utf-8") as f:
        for pred in predictions:
            f.write(json.dumps(pred, ensure_ascii=False) + "\n")
    print(f"\n✅ Predictions saved to {pred_file}")
    print(f"\nNext: python pipeline/evaluate.py --eval-dataset data/val.jsonl --predictions data/val-predictions.jsonl")


# ---------------------------------------------------------------------------
# Subcommand: list-models
# ---------------------------------------------------------------------------


def cmd_list_models(_args: argparse.Namespace) -> None:
    """List all fine-tunable models available on Together.ai."""
    print("Known fine-tunable models (hardcoded):")
    for m in FINE_TUNABLE_MODELS:
        print(f"  - {m}")

    print("\nFetching from API...")
    try:
        data = _api_get("/models")
        model_ids = [m["id"] for m in data if isinstance(m, dict) and "id" in m]
        print(f"  Found {len(model_ids)} models total on Together.ai")
        # Show fine-tuning compatible ones
        for m in FINE_TUNABLE_MODELS:
            status = "✅ in API" if m in model_ids else "❌ not in API"
            print(f"  - {m} {status}")
    except Exception as e:
        print(f"  API error: {e}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Together.ai fine-tuning client (validate/train/infer)",
    )
    subparsers = parser.add_subparsers(dest="command", help="Subcommand")

    # --- validate ---
    p_validate = subparsers.add_parser("validate", help="Validate JSONL + split train/val")
    p_validate.add_argument("--input", "-i", required=True, help="Input JSONL file")
    p_validate.add_argument("--system", "-s", default=None, help="System prompt to inject")
    p_validate.add_argument("--split-ratio", type=float, default=0.8, help="Train split ratio (default: 0.8)")
    p_validate.add_argument("--seed", type=int, default=42, help="Random seed for splitting")
    p_validate.add_argument("--dry-run", action="store_true", help="Validate without writing files")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Upload + start fine-tuning")
    p_train.add_argument("--model", "-m", default=None, help="Model ID (default: Qwen/Qwen2.5-7B-Instruct)")
    p_train.add_argument("--lr", type=float, default=DEFAULT_LR, help="Learning rate multiplier")
    p_train.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS, help="Number of epochs")
    p_train.add_argument("--timeout", type=int, default=DEFAULT_TIMEOUT, help="Poll timeout in seconds")

    # --- infer ---
    p_infer = subparsers.add_parser("infer", help="Inference on val examples")
    p_infer.add_argument("--model-id", default=None, help="Model ID override (reads .cache/model_id.txt by default)")

    # --- list-models ---
    subparsers.add_parser("list-models", help="List fine-tunable models")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    if args.command == "validate":
        cmd_validate(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "infer":
        cmd_infer(args)
    elif args.command == "list-models":
        cmd_list_models(args)


if __name__ == "__main__":
    main()
