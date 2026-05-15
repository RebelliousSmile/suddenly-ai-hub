# 🎭 Suddenly AI Hub

> **Fine-tuned French roleplay models for Suddenly.**

## What this project is

LoRA adapters fine-tuned on French tabletop RP data (Discord logs, text RP sessions, RP forums). Trained to help Suddenly's AI features understand and generate RP-appropriate content.

## Training data

The models are fine-tuned on:
- **Discord RP logs** — actual French text RP from RPDiscord, RPDiscord2, RPDiscord3
- **Forum RP** — La Cour d'Obéron, RP Discord logs
- **Ren'Py dialogues** — extracted dialogues from French text adventure games
- **Google Books** — French RP fiction excerpts
- **Playwright scraping** — RP-related content

## Model Registry

Browse available adapters:

```bash
python scripts/list_models.py
```

The models are fine-tuned for **Suddenly's AI features**:

### 1. Suggestion de dialogue (#77)

Generate dialogue that fits a character's personality and the scene context.

```bash
python scripts/infer.py \
  --adapter models/suddenly-dialogue \
  --prompt "Contexte: Tavern scene. Personnage: marchand cynique. \
Prompt: {personnage} répond aux joueurs..."
```

### 2. Suggestion d'action (#78)

Propose plausible actions for a character in a given situation.

```bash
python scripts/infer.py \
  --adapter models/suddenly-action \
  --prompt "Contexte: Combat en forêt. Personnage: elfe archer. \
Prompt: {personnage} veut..."
```

### 3. Suggestion de description (#79)

Generate atmospheric scene descriptions matching the RP tone.

```bash
python scripts/infer.py \
  --adapter models/suddenly-description \
  --prompt "Contexte: Ruines maudites. Ambiance: sombre, mystérieuse. \
Prompt: Décrire la salle principale..."
```

### 4. Suggestion de pensée intérieure (#80)

Generate internal monologue that matches character psychology.

```bash
python scripts/infer.py \
  --adapter models/suddenly-thought \
  --prompt "Contexte: Le personnage trahit son allié. \
Prompt: {personnage} pense en silence..."
```

### 5. Analyse de cohérence RP (#81-#82)

Check RP consistency at scene or session level.

```bash
python scripts/infer.py \
  --adapter models/suddenly-consistency \
  --prompt "Scène: [liste des reports]. \
Analyser la cohérence des actions de {personnage}..."
```

### 6. Résumé de session (#83)

Generate session summaries for archive and recaps.

```bash
python scripts/infer.py \
  --adapter models/suddenly-summary \
  --prompt "Session #42. Reports: [liste complète]. \
Résumer cette session en 500 mots..."
```

### 7. Suggestion de liens fédérés (#84)

Suggest claim/adopt/fork links via AI analysis.

```bash
python scripts/infer.py \
  --adapter models/suddenly-federation \
  --prompt "Analyze these reports and suggest federation links..."
```

## Using adapters

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

# Load Suddenly LoRA adapter
adapter_path = "models/suddenly-dialogue"
model = PeftModel.from_pretrained(model, adapter_path)
```

## What you can build

| Use case | Description |
|----------|-------------|
| **Suddenly AI features** | Direct integration with Suddenly's LLMClient |
| **RP assistance bots** | Discord/Telegram bots for suggestion features |
| **Campaign tools** | Scene descriptions, NPC dialogue, session summaries |
| **RP analysis** | Consistency checking, character tracking |

## License

MIT — use for anything, commercial projects welcome.

## Contact

[RebelliousSmile](https://github.com/RebelliousSmile) — [GitHub](https://github.com/RebelliousSmile/suddenly-ai-hub)
