# 🎭 Suddenly AI Hub

> **Fine-tuned French roleplay models for Suddenly.**

## What this project is

LoRA adapters fine-tuned on French tabletop RP data (Discord logs, text RP sessions, RP forums). Trained to help Suddenly's AI features understand and generate RP-appropriate content.

The models use a **stacked LoRA architecture** along three orthogonal axes:

| Axe | What it controls | Examples |
|---|---|---|
| **Univers** | The world — genre, lore, magic/tech systems | `fantasy-medievale`, `cyberpunk`, `steampunk`, `horreur-gothique` |
| **Situation** | Tone and rhythm — the emotional/temporal quality of the scene | `combat`, `romance`, `intrigue`, `politique`, `quotidien` |
| **Voix** | The GM's narrative voice — stylistic personality | `solennel`, `narquois`, `theatral`, `neutre`, `lyrique` |

At inference time, adapters from each axis are combined via **stacking** — a single API call with distinct multipliers per adapter. The three dimensions are independent: any univers can be paired with any situation and any voice.

## Model Registry

Browse available adapters:

```bash
python scripts/list_models.py
```

## Architecture

### Three orthogonal axes

**Axe 1 — Univers** (genre/lore)

| Adapter | Description |
|---|---|
| `fantasy-medievale` | Épées, magie, féodalité, royaumes |
| `cyberpunk` | Techno, mégacorporations, implants |
| `steampunk` | Vapeur, engrenages, époque victorienne |
| `horreur-gothique` | Vampires, atmosphère sombre, décennie sombre |

**Axe 2 — Situation** (ton/rythme de la scène)

| Adapter | Description |
|---|---|
| `combat` | Escarmouches, batailles, tensions physiques |
| `romance` | Relations interpersonnelles, tension émotionnelle |
| `intrigue` | Manœuvres, trahisons, mystères, manipulations |
| `politique` | Négociations, alliances, diplomatie, enjeux sociaux |
| `quotidien` | Moments de repos, interactions sociales légères |

**Axe 3 — Voix** (personnalité narrative du MJ)

| Adapter | Description |
|---|---|
| `solennel` | Ton grave, solennel, épique |
| `narquois` | Ironique, pince-sans-rire, espiègle |
| `theatral` | Spectaculaire, emphatique, dramatique |
| `neutre` | Sobre, direct, non-intrusif |
| `lyrique` | Poétique, descriptif, sensuel |

### Stacking

At inference time, one adapter per axis is combined:

```
W_final = W_base + α₁·(A₁×B₁)_univers + α₂·(A₂×B₂)_situation + α₃·(A₃×B₃)_voix
```

Together AI natively supports passing **multiple adapter IDs** with **separate multipliers** in a single API call. The model weights add linearly at inference.

### Multipliers: calibrating intensity

| Configuration | Univers | Situation | Voix | Result |
|---|---|---|---|---|
| Équilibré | 1.0 | 1.0 | 1.0 | Balance univers + scène + style |
| Univers dominant | 1.5 | 0.8 | 0.8 | Lore fort, ton et voix subtils |
| Ton dominant | 0.8 | 1.5 | 1.0 | Scène marquée, voix expressive |
| Voix seule | 0.0 | 1.0 | 1.5 | Pas de lore, ton + voix forts |
| Univers seul | 1.0 | 0.0 | 0.0 | Lore fort, sans spécialisation de scène/voix |

> **Warning:** Multipliers > 2.0 risk catastrophic forgetting of the base model. Stick to 0.0–2.0.

## Using adapters

### Together AI API (recommended)

```bash
curl -s https://api.together.xyz/v1/chat/completions \
  -H "Authorization: Bearer $TOGETHER_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen2.5-7B-Instruct",
    "messages": [
      {"role": "system", "content": "Vous êtes le MJ d\'une campagne de jeu de rôle français."},
      {"role": "user", "content": "Le marchand cynique sort son épée..."}
    ],
    "adapters": [
      {"id": "suddenly/fantasy-medievale", "multiplier": 1.0},
      {"id": "suddenly/combat", "multiplier": 1.0},
      {"id": "suddenly/narquois", "multiplier": 1.0}
    ],
    "max_tokens": 300
  }'
```

### Local inference with vLLM (pre-merged)

```python
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load base model
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
model = AutoModelForCausalLM.from_pretrained(
    "Qwen/Qwen2.5-7B-Instruct",
    torch_dtype="auto",
    device_map="auto"
)

# Load stacked LoRA adapters
model = PeftModel.from_pretrained(model, "models/fantasy-medievale")
model = PeftModel.from_pretrained(model, "models/combat", adapter_name="situation")
model = PeftModel.from_pretrained(model, "models/narquois", adapter_name="voix")

# Generate with combined adapters
model.set_adapter(["fantasy-medievale", "situation", "voix"])
```

### Local inference script

```bash
# Single adapter
python scripts/infer.py --adapter fantasy-medievale --prompt "Le château se dresse sur la colline..."

# Stacked inference (multi-adapter)
python scripts/infer.py --stack \
  --adapter-1 fantasy-medievale --multiplier-1 1.0 \
  --adapter-2 combat --multiplier-2 1.0 \
  --adapter-3 narquois --multiplier-3 1.0 \
  --prompt "Le marchand cynique sort son épée..."
```

## What you can build

| Use case | Description |
|----------|-------------|
| **Suddenly AI features** | Direct integration with Suddenly's LLMClient |
| **RP assistance bots** | Discord/Telegram bots for scene generation |
| **Campaign tools** | Scene descriptions, NPC dialogue, session summaries |
| **RP analysis** | Consistency checking, character tracking |

## Fallback hierarchy

When a stacked combination is unavailable:

1. Full stack (univers + situation + voix)
2. Two-axis stack (univers + situation)
3. Single adapter (situation or univers)
4. Generic French RP LoRA
5. Base model only

## Evaluation

A complete evaluation pipeline is available to compare stacked models against each other and against the base model.

### Dataset

50 test prompts (`data/test-prompts.jsonl`) covering:

| Dimension | Values |
|-----------|--------|
| **Univers** | fantasy-medievale, cyberpunk |
| **Situation** | combat, romance, intrigue, politique, quotidien |
| **Voix** | solennel, narquois, theatral, neutre, lyrique |

Each univers×situation combination has 5 prompts (one per voice), split across description and dialogue categories.

### Running evaluation

```bash
# Evaluate a stacked combination
python scripts/evaluate.py --stack \
  --adapter-1 fantasy-medievale --multiplier-1 1.0 \
  --adapter-2 combat --multiplier-2 1.0 \
  --adapter-3 narquois --multiplier-3 1.0

# Quick combination sweep (sample output for first 2 combos)
python scripts/evaluate.py --full

# Baseline: base model without any LoRA
python scripts/baseline.py --test-data data/test-prompts.jsonl
```

### Evaluation dimensions

Each output is scored on:

| Dimension | What it measures |
|-----------|-----------------|
| **relevance** | Does the output match the prompt intent? |
| **style** | Does it match the requested style/voice? |
| **immersion** | Does it feel like authentic RP? |
| **creativity** | Is it interesting and non-generic? |
| **length_score** | Is the response adequately detailed? |

### Tests

```bash
python -m pytest tests/test_suddenly_muses.py -v
```

Validates data integrity, script structure, stacking configuration, and README content.

## License

MIT — use for anything, commercial projects welcome.

## Contact

[RebelliousSmile](https://github.com/RebelliousSmile) — [GitHub](https://github.com/RebelliousSmile/suddenly-muses)
