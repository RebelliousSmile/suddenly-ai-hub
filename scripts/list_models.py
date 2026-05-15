#!/usr/bin/env python3
"""List all available LoRA adapters in the models/ directory."""
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# 3 independent dimensions
UNIVERSES = {
    "cyberpunk": "Neon noir, corporate intrigue, gritty tech",
    "fantasy": "Medieval magic, dungeons, epic quests",
    "horror": "Lovecraftian dread, psychological tension",
    "scifi": "Space opera, dystopian, AI themes",
    "seinen": "Mature themes, complex moral situations",
}

NARRATION_TYPES = {
    "combat": "Fast-paced, visceral, tactical descriptions",
    "romance": "Tension, intimacy, emotional depth",
    "intrigue": "Political maneuvering, hidden agendas, dialogue-heavy",
    "exploration": "Wonder, discovery, world-building, sensory detail",
    "dialogue": "Natural conversation, personality, wit",
    "drama": "Character depth, moral complexity, consequences",
}

ROLES = {
    "dm": "Dungeon Master — scene description + NPC dialogue + action management",
    "npc": "Single character — specific personality, consistent voice",
    "narrator": "Pure storytelling — descriptive, atmospheric, third-person",
}

NPC_NAMES = {
    "cyberpunk": "merchant",
    "fantasy": "village",
    "horror": "survivor",
    "scifi": "officer",
    "seinen": "mentor",
}


def main():
    print("🎭 Suddenly AI Hub — Available LoRA Adapters")
    print("=" * 60)
    print()

    # Show available adapters
    print("Available adapters:")
    print()

    for universe in sorted(UNIVERSES):
        print(f"  🌐 {universe.upper()} — {UNIVERSES[universe]}")
        print()

        for role in sorted(ROLES):
            npc_tag = f"-{NPC_NAMES[universe]}" if role == "npc" else ""
            adapter_base = f"{universe}-{role}{npc_tag}"
            adapter_path = MODELS_DIR / adapter_base
            status = "✅" if adapter_path.exists() else "📋"
            print(f"     {status} {adapter_base:40s} — {ROLES[role]}")

        print()

    # Show combination matrix
    print("Combination matrix:")
    print()
    print("  Each model can be combined with narration types:")
    for ntype, ndesc in sorted(NARRATION_TYPES.items()):
        print(f"    • {ntype:15s} — {ndesc}")
    print()

    print("Usage:")
    print("  python scripts/infer.py --adapter <universe-role> --prompt '...'\n")
    print("  from peft import PeftModel")
    print("  model = PeftModel.from_pretrained(base_model, 'models/<universe-role>/')")


if __name__ == "__main__":
    main()
