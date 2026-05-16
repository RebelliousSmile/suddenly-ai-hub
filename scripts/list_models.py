#!/usr/bin/env python3
"""List all available LoRA adapters in the models/ directory."""
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

# Suddenly AI — 3 axes stacking

AXES = {
    "Axe 1 — Univers (genre/lore)": {
        "suddenly-fantasy-medievale": {"desc": "Épées, magie, féodalité, royaumes"},
        "suddenly-cyberpunk": {"desc": "Techno, mégacorporations, implants"},
        "suddenly-steampunk": {"desc": "Vapeur, engrenages, époque victorienne"},
        "suddenly-horreur-gothique": {"desc": "Vampires, atmosphère sombre"},
    },
    "Axe 2 — Situation (ton/rythme)": {
        "suddenly-combat": {"desc": "Escarmouches, batailles, tensions physiques"},
        "suddenly-romance": {"desc": "Relations interpersonnelles, tension émotionnelle"},
        "suddenly-intrigue": {"desc": "Manœuvres politiques, trahisons, mystères"},
        "suddenly-politique": {"desc": "Négociations, alliances, diplomatie"},
        "suddenly-quotidien": {"desc": "Moments de repos, interactions sociales légères"},
    },
    "Axe 3 — Voix (personnalité narrative)": {
        "suddenly-solennel": {"desc": "Ton grave, solennel, épique"},
        "suddenly-narquois": {"desc": "Ironique, pince-sans-rire, espiègle"},
        "suddenly-theatral": {"desc": "Spectaculaire, emphatique, dramatique"},
        "suddenly-neutre": {"desc": "Sobre, direct, non-intrusif"},
        "suddenly-lyrique": {"desc": "Poétique, descriptif, sensuel"},
    },
}


def main():
    print("🎭 Suddenly AI Hub — Available LoRA Adapters")
    print("=" * 70)
    print()
    print("  3-axis stacking: Univers × Situation × Voix")
    print(f"  Combinations: {sum(len(v) for v in AXES.values())} adapters")
    print()

    for axis_name, adapters in AXES.items():
        print(f"{'═' * 70}")
        print(f"  {axis_name}")
        print(f"{'─' * 70}")
        print(f"  {'Adapter':32s}  Status  Description")
        for adapter_id, info in sorted(adapters.items()):
            adapter_path = MODELS_DIR / adapter_id
            status = "✅" if adapter_path.exists() else "📋"
            print(f"  {adapter_id:32s}  {status}  {info['desc']}")
        print()

    print(f"{'═' * 70}")
    print()
    print("Usage — Stacked inference:")
    print("  python scripts/infer.py --stack \\")
    print("    --adapter-1 fantasy-medievale --multiplier-1 1.0 \\")
    print("    --adapter-2 combat           --multiplier-2 1.0 \\")
    print("    --adapter-3 narquois         --multiplier-3 1.0 \\")
    print("    --prompt 'Le marchand sort son épée...'")
    print()
    print("Usage — Single adapter:")
    print("  python scripts/infer.py --adapter fantasy-medievale --prompt '...'")
    print()
    print("  from peft import PeftModel")
    print("  model = PeftModel.from_pretrained(base, 'models/<adapter>/')")


if __name__ == "__main__":
    main()
