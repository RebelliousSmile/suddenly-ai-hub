# 🎭 Suddenly AI Hub

> **Fine-tuned French roleplay models — ready to use.**

## What this project is

Trained language models specialized in French roleplay. They understand scene descriptions, NPC dialogue, and player actions natively — without complex prompting.

## Quick Start

```python
from transformers import AutoModelForCausalLM, AutoTokenizer, LoraConfig, get_linear_schedule_with_warmup, BitsAndBytesConfig
import torch

# 1. Load base model + LoRA adapter for your chosen universe
model_id = "suddenly-ai-hub/cyberpunk-dm"  # see model registry below
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.float16,
    device_map="auto",
    use_auth_token=False  # local path or HF token
)

# 2. Run inference
prompt = "Le groupe arrive devant la porte enchantée."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

## Model Registry

All trained models are LoRA adapters fine-tuned on French RP dialogues. Browse available models:

```bash
python scripts/list_models.py
```

**Available universes & roles:**

| Universe | Roles (LoRA adapters) | Description |
|----------|----------------------|-------------|
| **cyberpunk** | `dm`, `npc_merchant`, `narrator` | Neon noir, corporate intrigue |
| **fantasy** | `dm`, `npc_village`, `narrator` | Medieval, magic, dungeons |
| **horror** | `dm`, `npc_survivor`, `narrator` | Lovecraftian, psychological, dark |
| **scifi** | `dm`, `npc_officer`, `narrator` | Space opera, dystopian, AI |
| **seinen** | `dm`, `npc_mentor`, `narrator` | Mature themes, complex drama |

Each universe comes with 3 LoRA variants:
- **`dm`** — Dungeon Master style (scene description + NPC dialogue + action management)
- **`npc_*`** — Single character roleplay (specific personality, consistent voice)
- **`narrator`** — Pure storytelling (descriptive, atmospheric, third-person)

## Using models in your project

### Swap universes at runtime

```python
# Load base model once, then swap LoRA adapters
from peft import PeftModel

base_model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct", torch_dtype="auto", device_map="auto")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Switch to cyberpunk DM
base_model = PeftModel.from_pretrained(base_model, "models/cyberpunk-dma-adapter/")
# Switch to fantasy NPC
base_model = PeftModel.from_pretrained(base_model, "models/fantasy-npc-village-adapter/")
```

### Context-aware prompting

The model is trained to respect the role context:

```
## Cyberpunk DM
Context: Year 2089, Neo-Tokyo district 7. Corporate guards patrol the rain-slicked streets.
Player: "{user_input}"

## Fantasy NPC
Context: You are a cynical tavern keeper in a medieval fantasy setting. You sell information, not weapons.
Player: "{user_input}"

## Horror Narrator
Context: Describe a scene with sensory details, slow pacing, and mounting dread.
Player: "{user_input}"
```

### Local CLI

```bash
# Quick inference
python scripts/infer.py --adapter models/cyberpunk-dm --prompt "Le groupe arrive devant la porte..."

# List available adapters
python scripts/list_models.py
```

## What you can build

| Use case | Description |
|----------|-------------|
| **D&D bots** | Discord/Telegram bots that act as GMs or NPCs |
| **Tabletop companion** | Real-time scene narration, NPC responses |
| **Interactive fiction** | Ren'Py, Twine, or custom game engines |
| **Campaign prep** | Generate encounters, NPCs, descriptions |

## License

MIT — use for anything, commercial projects welcome.

## Contact

[RebelliousSmile](https://github.com/RebelliousSmile) — [GitHub](https://github.com/RebelliousSmile/suddenly-ai-hub)
