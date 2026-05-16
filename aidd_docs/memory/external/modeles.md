| Modèle | Taille | Spécialisation | NSFW | Lien | Fine-tunable sur Together ? |
| --- | --- | --- | --- | --- | --- |
| **Qwen2.5-7B-Instruct** | 7B | RP, français, dialogue | ✅ Oui | [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) | ✅ **RECOMMANDÉ** |
| **Llama-3.1-8B-Instruct** | 8B | Généraliste | ❌ Non | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-8B-Instruct) | ✅ |
| **Llama-3.1-70B-Instruct-Turbo** | 70B | Plus puissant | ❌ Non | [Hugging Face](https://huggingface.co/meta-llama/Llama-3.1-70B-Instruct) | ✅ (fallback) |
| **Mistral-7B-Instruct** | 7B | Généraliste | ❌ Non | [Hugging Face](https://huggingface.co/mistralai/Mistral-7B-Instruct) | ⚠️ Non retenu (Qwen2.5 meilleur en français) |
| **Mixtral-8x24B-Instruct** | 47B | Élevé débit | ❌ Non | [Hugging Face](https://huggingface.co/mistralai/Mixtral-8x24B-Instruct) | ⚠️ Fallback — non retenu |

## Notes

### ❌ Modèles locaux (HuggingFace uniquement — pas sur Together.ai)
Ces modèles sont fine-tunables **localement** (Ollama, VLLM), mais **PAS** sur Together.ai cloud :

| Modèle | Taille | Spécialisation | NSFW | Lien |
| --- | --- | --- | --- | --- |
| Pygmalion-13B | 13B | Roleplay, créativité | ✅ Oui | [Hugging Face](https://huggingface.co/PygmalionAI/pygmalion-13b) |
| Metharme-7B | 7B | Roleplay, NSFW | ✅ Oui | [Hugging Face](https://huggingface.co/Metharme/Metharme-7B) |
| Nous-Hermes-2-Mistral | 7B | Roleplay, créativité | ❌ Non | [Hugging Face](https://huggingface.co/NousResearch/Nous-Hermes-2-Mistral-7B) |
| OpenChat-3.5 | 7B | Dialogues ouverts | ❌ Non | [Hugging Face](https://huggingface.co/openchat/openchat-3.5-7b) |
| StableLM-3B | 3B | Léger, adaptable | ❌ Non | [Hugging Face](https://huggingface.co/stabilityai/stablelm-3b) |

### 🎯 Choix pour Suddenly

**Modèle principal : `Qwen/Qwen2.5-7B-Instruct`**
- Excellent français (bien supérieur à Llama/Mistral)
- Optimisé pour le dialogue (Instruct variant)
- Coût bas (~$2.70/M tokens)
- Disponible en fine-tuning sur Together.ai

**Fallback :** `Llama-3.1-70B-Instruct-Turbo` si Qwen ne marche pas en fine-tuning.
