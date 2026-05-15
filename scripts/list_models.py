#!/usr/bin/env python3
"""List all available LoRA adapters in the models/ directory."""
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Suddenly AI features — from issues #76 to #84
FEATURES = {
    "suddenly-dialogue": {
        "issue": "#77",
        "desc": "Suggestion de dialogue pour un personnage",
    },
    "suddenly-action": {
        "issue": "#78",
        "desc": "Suggestion d'action pour un personnage",
    },
    "suddenly-description": {
        "issue": "#79",
        "desc": "Suggestion de description de scène",
    },
    "suddenly-thought": {
        "issue": "#80",
        "desc": "Suggestion de pensée intérieure",
    },
    "suddenly-consistency-scene": {
        "issue": "#81",
        "desc": "Analyse de cohérence RP sur une scène",
    },
    "suddenly-consistency-session": {
        "issue": "#82",
        "desc": "Analyse de cohérence RP sur toute la session",
    },
    "suddenly-summary": {
        "issue": "#83",
        "desc": "Génération automatique du résumé de session",
    },
    "suddenly-federation": {
        "issue": "#84",
        "desc": "Suggestions de liens claim/adopt/fork via IA",
    },
}


def main():
    print("🎭 Suddenly AI Hub — Available LoRA Adapters")
    print("=" * 70)
    print()
    print(f"{'Adapter':30s} {'Status':10s}  Issue  Feature")
    print("-" * 70)

    for adapter_id, info in sorted(FEATURES.items()):
        adapter_path = MODELS_DIR / adapter_id
        status = "✅" if adapter_path.exists() else "📋"
        print(f"  {adapter_id:30s} {status:10s}  {info['issue']}  {info['desc']}")

    print()
    print("Usage:")
    print("  python scripts/infer.py --adapter <adapter_id> --prompt '...'")
    print()
    print("  from peft import PeftModel")
    print("  model = PeftModel.from_pretrained(base_model, 'models/<adapter_id>/')")


if __name__ == "__main__":
    main()
