# 🎭 Suddenly AI Hub

> **Fine-tuned French roleplay models — ready to use.**

## What this project is

Trained language models specialized in French roleplay. They understand scene descriptions, NPC dialogue, and player actions natively — without complex prompting.

## Using the models

### Python

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained(
    "models/your-model",
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained("models/your-model")

inputs = tokenizer("Le groupe arrive devant la porte enchantée.", return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Local inference

```bash
# Run the model locally
python scripts/infer.py --checkpoint models/your-model --prompt "Tu es un PNJ..."

# REST API mode
python scripts/api.py --checkpoint models/your-model --port 8080
```

### System prompt patterns

The model works best with role-specific prompts:

```
## Maître de jeu (DM)
Tu es un maître de jeu. Tu décris les scènes, incarnes les PNJ et gères l'action.
Le joueur dit: "{user_input}"

## PNJ
Tu es un marchand cynique dans une taverne cyberpunk. Tu vends des informations, pas des armes.
Le joueur dit: "{user_input}"

## Narrateur
Tu racontes une scène de jeu. Style sombre, descriptions sensorielles, rythme tendu.
Le joueur dit: "{user_input}"
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
