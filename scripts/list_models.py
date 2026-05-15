#!/usr/bin/env python3
"""List all available LoRA adapters in the models/ directory."""
from pathlib import Path
import json

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Universe/role registry
REGISTRY = {
    "cyberpunk": {
        "dm": "Dungeon Master — scene description + NPC dialogue + action management",
        "npc_merchant": "Cynical information broker in a neon-lit bazaar",
        "narrator": "Noir storyteller, atmospheric and gritty",
    },
    "fantasy": {
        "dm": "Dungeon Master — medieval fantasy, magic, dungeons",
        "npc_village": "Tavern keeper, village elder, or guild master",
        "narrator": "Epic fantasy narrator, mythic tone",
    },
    "horror": {
        "dm": "Horror DM — Lovecraftian dread, psychological tension",
        "npc_survivor": "Paranoid survivor, fragmented memories",
        "narrator": "Atmospheric horror, slow-burn tension",
    },
    "scifi": {
        "dm": "Sci-fi DM — space opera, dystopian, AI themes",
        "npc_officer": "Stellar fleet officer, by-the-book but pragmatic",
        "narrator": "Hard sci-fi narrator, clinical yet poetic",
    },
    "seinen": {
        "dm": "Seinen DM — mature themes, complex moral situations",
        "npc_mentor": "Wise but flawed mentor figure",
        "narrator": "Literary narrator, introspective and nuanced",
    },
}


def main():
    print("🎭 Suddenly AI Hub — Available LoRA Adapters")
    print("=" * 60)
    print()

    for universe, roles in sorted(REGISTRY.items()):
        print(f"  🌐 {universe.upper()}")
        for role, desc in sorted(roles.items()):
            model_id = f"{universe}-{role}"
            adapter_path = MODELS_DIR / model_id
            status = "✅" if adapter_path.exists() else "📋"
            print(f"     {status} {model_id:30s} — {desc}")
        print()

    print("Usage:")
    print("  python scripts/infer.py --adapter <universe-role> --prompt '...'\n")
    print("  from peft import PeftModel")
    print("  model = PeftModel.from_pretrained(base_model, 'models/<universe-role>/')")


if __name__ == "__main__":
    main()
