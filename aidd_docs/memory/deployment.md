---
name: deployment
description: Infrastructure and deployment documentation
---

# Deployment

## Infrastructure Costs

| Component | Solution | Estimated Cost |
|-----------|----------|----------------|
| API Gateway | Hetzner VPS CX21 | ~5€/month |
| Database | PostgreSQL on Supabase or VPS | 0-10€/month |
| Storage | Backblaze B2 or Hetzner Object Storage | ~1€/month per 100GB |
| Inference GPU | RunPod RTX 3090 (spot) | ~0.20$/h, ~50$/month moderate load |
| Fine-tuning | Together.ai Fine-tuning API → Axolotl + RunPod | 50-200$/run |
| **Total** | **Steady state** | **~80-150€/month** |

## GPU Scaling

- Low traffic: minimal instance RTX 3090 / A10 (~0.20$/h)
- Peak: additional worker on demand via RunPod API
- Saturated fallback: `retry-after` header + UI "unavailable" message

## URLs

- Production: `https://ai.suddenly.social`
- Transparency page: `https://ai.suddenly.social/transparency`

## Environments

- Phase 0 (beta closed): RunPod
- Phase 1+: production at `ai.suddenly.social`

## Model Deployment Process

1. Export corpus since last fine-tuning date
2. Submit to Together.ai Fine-tuning API
3. Evaluate on reserved test set
4. If metrics OK → progressive rollout (10% → 50% → 100%)
5. Archive previous model (rollback possible within 48h)

## Data Pipeline

1. Session submitted
2. Structural validation (format, language, minimum length)
3. Anonymization (NER → generic tokens, FR/EN)
4. Format as training examples (reports → prompt/completion pairs)
5. Storage in corpus (PostgreSQL + S3-compatible)
6. Marked "available for next fine-tuning"

## CI/CD Pipeline

N/A - not configured (Phase 0)

## Rollback

- 48h window after each model deployment
- Archive previous model kept for rollback
