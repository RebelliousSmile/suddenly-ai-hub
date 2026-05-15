# 🎭 Suddenly AI Hub

> **Fine-tuned language models for French roleplay — ready to use, ready to build on.**

## What this project does

You get **trained French RP chat models** out of the box. The repo contains the full training pipeline (data collection → corpus preparation → training → evaluation), but at the end of the day, the value is in the **models themselves**.

## Models

Trained models and checkpoints are stored in `models/` (or linked from external storage). To use a fine-tuned model:

### Quick inference with Transformers

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "models/your-fine-tuned-model"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)

prompt = "Tu es un maître de jeu dans un univers cyberpunk. Le joueur arrive dans un bar sombre..."
inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
outputs = model.generate(**inputs, max_new_tokens=512, temperature=0.8)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))
```

### Inference with Ollama (local, no GPU needed for small models)

```bash
# Pull the model (if on Ollama Hub)
ollama pull suddenly-ai-hub:latest

# Chat
ollama run suddenly-ai-hub

# Or API mode
ollama serve  # runs on :11434
curl http://localhost:11434/api/generate -d '{
  "model": "suddenly-ai-hub",
  "prompt": "Tu es un PNJ dans un donjon...",
  "stream": false
}'
```

### Via Together.ai API (cloud inference)

```bash
pip install together

python scripts/train_together.py \
  --model-path models/fine-tuned-checkpoint \
  --mode inference \
  --system-prompt "Tu es un maître de jeu..." \
  --messages '[{"role":"user","content":"J\'entre dans la taverne."}]'
```

## Using fine-tuned models in your projects

### As a chat system prompt replacement

The model is fine-tuned to speak in a specific RP register — no system prompt engineering needed:

```python
# Before: generic LLM, needs careful prompting
# "You are a D20 Dungeon Master in a dark fantasy setting..."

# After: the model IS the DM
response = model.generate(
    prompt="Le groupe arrive devant la porte enchantée.",
    max_new_tokens=300,
    temperature=0.7
)
```

### As a dialogue engine for game servers

The model understands scene descriptions, NPC dialogue, and player actions natively. It can power:

- **D&D/JDR bots** on Discord/Telegram
- **Tabletop companion tools** (scene narration, NPC responses)
- **Interactive fiction engines** (Ren'Py, Twine, custom)
- **Campaign preparation** (generating encounters, NPCs, descriptions)

### System prompt recipes

The model works best with context-aware prompts:

```
## Maître de jeu (DM)
Tu es un maître de jeu. Tu décris les scènes, incarnes les PNJ et gères l'action.
Le joueur dit: "{user_input}"

## PNJ
Tu es un marchand cynique dans une taverne cyberpunk. Tu vend des informations, pas des armes.
Le joueur dit: "{user_input}"

## Narrateur
Tu racontes une scène de jeu. Style sombre, descriptions sensorielles, rythme tendu.
Le joueur dit: "{user_input}"
```

## Training pipeline overview

If you want to train your own models, the pipeline is:

```
data/                          # Raw collected dialogues
  ├── renpy-corpus.jsonl       # Synthetic French corpus (5 genres)
  ├── train.jsonl              # Training set (Axolotl format)
  └── val.jsonl                # Validation set

scripts/crawl_rpv/             # Data collection tools
  ├── generate_corpus.py       # Generate synthetic RP dialogues
  ├── extract_real_dialogues.py # Extract from Ren'Py VN repos
  └── github_search.py         # Find VNs on GitHub

scripts/train_together.py      # Training & inference script
```

### Training

```bash
# Fine-tune via Together.ai (SFT)
python scripts/train_together.py \
  --mode train \
  --train-file data/train.jsonl \
  --val-file data/val.jsonl \
  --base-model Qwen/Qwen2.5-7B-Instruct \
  --output-dir models/output/

# Or with Axolotl (local GPU)
axolotl train configs/qwen2.5_7b_rp.yaml
```

### Corpus composition

The training data combines:

| Source | Type | Languages | Size |
|--------|------|-----------|------|
| Synthetic Ren'Py | Scene + dialogue pairs | French | ~50 entries, 5 genres |
| Ren'Py VNs (GitHub) | Real VN dialogues | English + FR translation | ~10k+ dialogues |
| RP forum scraping | Forum-style roleplay | French | Pending |

## Data format

All training data follows the **Axolotl JSONL format**:

```jsonl
{"messages": [{"role":"system","content":"Tu es un PNJ..."},{"role":"user","content":"Bonjour"},{"role":"assistant","content":"Salut, que cherches-tu..."}]}
```

## License

MIT — use the models for anything. Commercial projects welcome.

## Contact

- **Author**: [RebelliousSmile](https://github.com/RebelliousSmile)
- **GitHub**: [suddenly-ai-hub](https://github.com/RebelliousSmile/suddenly-ai-hub)
