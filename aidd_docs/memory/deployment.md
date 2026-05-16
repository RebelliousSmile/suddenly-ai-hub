---
name: deployment
description: Infrastructure and deployment documentation
---

# Deployment

## Infrastructure Costs

| Composant | Hébergement | Coût |
|-----------|-------------|------|
| Gateway + PostgreSQL | Railway | ~$5/mois fixe |
| Stockage corpus + modèles | Cloudflare R2 | 0€ (free tier 10GB) |
| Inférence GPU | RunPod spot (RTX 3090 / A10) | $0.20/h, scale to zero |
| Traitement data/fine-tuning | PC local (RTX 2080 Super) | 0€ |
| **Total standby** | | **~$5/mois** |
| **Total en charge modérée** | | **~$25-60/mois** |

## GPU Scaling

- **Standby** : pod éteint, 0$/h
- **Cold start** : ~2-3 min pour loader base + adapters depuis R2
- **Low traffic** : 1× RTX 3090 spot (~0.20$/h)
- **Peak** : scale-up via RunPod API (worker supplémentaire)
- **Saturated** : `retry-after` header + UI "Assistant indisponible, réessayez plus tard"

## Environments

- **Dev** : Gateway sur Railway (preview deploy), GPU = PC local avec Ollama/vLLM
- **Prod** : `https://ai.suddenly.social` sur Railway + RunPod spot

## Model Deployment Process

1. Fine-tuning sur PC local (Axolotl + RTX 2080 Super)
2. Export safetensors → upload Cloudflare R2
3. RunPod pull base + adapters depuis R2
4. Évaluer sur le set de test réservé
5. Si métriques OK → rollout progressif (10% → 50% → 100%)
6. Archive modèle précédent sur R2 (rollback possible dans 48h)

## Data Pipeline

1. Session soumise (contribute endpoint)
2. Validation structurelle (format, langue, longueur minimale)
3. Anonymisation (NER → tokens génériques, FR/EN)
4. Formatage en exemples d'entraînement (reports → paires prompt/completion)
5. Stockage sur Cloudflare R2 (JSONL)
6. Marqué "disponible pour le prochain fine-tuning"

## CI/CD Pipeline

N/A - not configured (Phase 0)

## Rollback

- 48h window after each model deployment
- Archive previous model kept for rollback
