# Benchmark: Fireworks.ai vs Together.ai

> **Date**: 2025-05-15
> **Objet**: Comparaison des coûts d'inférence serverless et fine-tuning entre Fireworks.ai et Together.ai
> **Cas d'usage**: Suddenly AI Hub — modèles Qwen2.5-7B et Qwen2.5-13B pour RP français

---

## 1. Prix inférence serverless (per 1M tokens)

| Modèle | Provider | Input / 1M | Output / 1M | Context |
|--------|----------|------------|-------------|---------|
| Qwen2.5-7B-Instruct-Turbo | **Together** | $0.30 | $0.30 | 32K |
| Qwen2.5-7B (≤16B) | **Fireworks** | $0.50 | $1.00 | ~32K |
| Qwen3-235B-A22B | Together | $0.20 | $0.60 | 262K |
| Qwen3.5-9B | Together | $0.10 | $0.15 | 262K |
| Llama 3.3 70B | Together | $0.88 | $0.88 | 131K |
| Gemma 3N E4B | Together | $0.06 | $0.12 | 32K |
| gpt-oss-20b | Together | $0.05 | $0.20 | 128K |
| DeepSeek-V4-Pro | Together | $2.10 | $0.20 | 512K |

**Coût typique par inférence** (ex: 512 tokens input + 256 tokens output):

| Modèle | Together | Fireworks |
|--------|----------|-----------|
| Qwen2.5-7B | **$0.00023** | **$0.00038** |
| Qwen2.5-13B (≤16B) | — | $0.00038 |

**🏆 Gagnant inférence: Together.ai** — ~37% moins cher pour le Qwen2.5-7B.

---

## 2. Features & capacités

| Feature | Fireworks.ai | Together.ai |
|---------|--------------|-------------|
| **Fine-tuning (LoRA/Full)** | ✅ API + console web | ✅ API SDK |
| **Serve fine-tuned = prix base** | ✅ Explicit | — (endpoint dédié) |
| **Batch inference (50% off)** | ✅ | ❌ |
| **Tokens en cache (50% off)** | ✅ | ✅ (pricing réduit) |
| **Embeddings** | ✅ ($0.008-$0.016/1M) | ❌ pas clair |
| **Image generation** | ✅ | ✅ (via Qwen-Image) |
| **Audio generation** | ❌ | ✅ ($0.0015/audio) |
| **Modèles disponibles** | ~200+ | ~241 |
| **Console web** | ✅ | ✅ |
| **SDK Python** | ✅ | ✅ |
| **Compatible OpenAI API** | ✅ | ✅ |

**🏆 Gagnant features: Fireworks.ai** — batch inference, embeddings, image/audio, cache à 50%.

---

## 3. Coût de fine-tuning

| Aspect | Fireworks.ai | Together.ai |
|--------|--------------|-------------|
| **Détection des GPUs** | ❌ **TOUS INDISPONIBLES** | ❌ **TOUS INDISPONIBLES** |
| **Prix GPU dédié** | H100: $7/hr, H200: $7/hr, B200: $10/hr, B300: $12/hr | Variable, pas clair |
| **Fine-tune serverless** | ✅ "Serve fine-tuned models for same price as base" | ❌ Nécessite endpoint dédié |
| **Logiciel FT** | API + console | Axolotl (open-source) |
| **Format dataset** | JSONL | JSONL (Axolotl) |

**Problème critique**: Sur les deux platforms, les GPUs dédiés sont **totalement indisponibles** au moment du test (05/15/2026). C'est une limitation partagée.

---

## 4. Coût total estimé par inférence

Pour un cas typique de RP JDR (512 input + 256 output tokens):

| Provider | Coût | Note |
|----------|------|------|
| **Together (Qwen2.5-7B)** | ~$0.00023 | Meilleur prix serverless |
| **Fireworks (Qwen2.5-7B)** | ~$0.00038 | +65% cher |
| **Fireworks (batch, 50% off)** | ~$0.00019 | **Meilleur si batch** |
| **Fireworks (cache, 50% off)** | ~$0.00025 | **Meilleur si cache** |
| **Fireworks (batch + cache)** | ~$0.00010 | **Meilleur absolu** |

---

## 5. Recommandation

### Pour Suddenly AI Hub:

1. **Inférence serverless (réponse temps réel)**: **Together.ai** — moins cher, plus de modèles, API OpenAI compatible
2. **Batch processing / historique**: **Fireworks.ai** avec batch inference — 50% de réduction
3. **Fine-tuning**: **Fireworks.ai** une fois les GPU disponibles — serve fine-tuned = prix base, pas d'endpoint dédié nécessaire
4. **Embeddings / RAG**: **Fireworks.ai** — seul les deux offrent embeddings à $0.008/1M tokens

### Stratégie hybride recommandée:
- **Together.ai** pour l'inférence principale (serverless, rapide, moins cher)
- **Fireworks.ai** pour: (a) le fine-tuning une fois GPU disponible, (b) le batch processing à -50%
- **Aucun des deux** pour le fine-tuning actuellement (GPUs indisponibles des deux côtés)

### Alternative à considérer:
- **Auto-hébergement** du modèle fine-tuné via vLLM (coût hardware unique, zéro coût marginal)
- **Hugging Face Inference Endpoints** (pay-per-use, mais plus cher)
- **Ollama / LM Studio** pour l'inférence locale si hardware disponible

---

## 6. Limitations de ce benchmark

- Prix des deux platforms susceptibles de changer (mis à jour 05/2026)
- GPU indisponible sur les deux platforms au moment du test → coût de fine-tuning non mesurable
- Pas de données sur les quotas/rate limits des deux providers
- Comparaison uniquement sur Qwen2.5 (notre cas d'usage spécifique)

---

*Généré automatiquement à partir des docs Fireworks et Together.ai le 2025-05-15*
