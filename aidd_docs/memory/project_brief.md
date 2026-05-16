---
name: project-brief
description: Project vision and domain documentation
---

# PROJECT_BRIEF.md

## Executive Summary
- **Project Name**: `suddenly-ai-hub`
- **Vision**: Communal AI inference hub for the Fediverse/ActivityPub ecosystem
- **Mission**: Fine-tune and host LLMs specialized in RP writing assistance for Suddenly federated instances

### Full Description
- Centralized ML infrastructure fed by federated data from participating Suddenly instances
- No dependency on commercial inference providers
- API REST propriétaire consommée par les instances Suddenly (pas de contrainte OpenAI SDK)
- ActivityPub-based auth with opt-in session contribution for corpus building

## Context

### Core Domain
- Federated roleplay writing platform (Suddenly) powered by specialized LLMs
- Hub handles fine-tuning, hosting, and inference — instances handle social and data
- Corpus built from opt-in user session contributions, anonymized before use
- Models never shared with commercial third parties; corpus stays private

### Ubiquitous Language

| Term | Definition | Synonyms |
|------|-----------|----------|
| Instance | Fediverse server running Suddenly | — |
| Hub | This centralized AI infrastructure | suddenly-ai-hub |
| Session | A RP roleplay session (source of training data) | — |
| Corpus | Collection of anonymized sessions for fine-tuning | — |
| Fine-tuning | Supervised training run to specialize the model | — |
| Contribution | Opt-in submission of a session to the corpus | — |
| Feature | AI capability exposed by the API | — |
| Inference | Generating text from the fine-tuned model | — |
| Gateway | FastAPI API entry point | — |

## Features & Use-cases

### API Endpoints
- `POST /v1/chat/completions` — inférence (convention de chemin, pas contrainte OpenAI)
- `GET /v1/models` — list available models + adapters actifs
- `GET /v1/health` — service health check
- `POST /v1/contribute` — opt-in session contribution (push)
- `GET /v1/stats` — usage and corpus stats

### Model Routing — deux dimensions complémentaires

**Par feature** (taille/contexte requis) :
- `suggest_short` → suddenly-7b-q4 (2k ctx)
- `suggest_dialogue`, `suggest_action`, `suggest_desc` → suddenly-7b (4k ctx)
- `suggest_thought`, `analyze_scene` → suddenly-7b (8k ctx)
- `analyze_session`, `generate_summary` → suddenly-13b (16k ctx)
- `suggest_links` → suddenly-13b (32k ctx)

**Par spécialisation** (LoRA adapter sélectionné via `genre` + `situation`) :
- `genre` : paramètre API côté client — correspond à l'**axe univers** GROG (medieval-fantastique, scifi, contemporain…)
- `situation` : paramètre API côté client — correspond à l'**axe situation** (combat, romance, enquête…)
- Les deux dimensions se combinent : feature choisit le modèle de base, LoRA spécialise le style

### Auth & Governance
- ActivityPub auth via instance public keys — no secrets stored
- Admins choose which features to activate per instance
- Users choose if their sessions contribute to corpus
- Instances can stop contributing without losing inference access
- Public transparency page at `https://ai.suddenly.social/transparency`

## Roadmap

### Phase 0 — Foundations (BLOCKS everything)
- Anonymization pipeline
- First fine-tune on public RP corpus
- ActivityPub auth implementation
- Deploy on RunPod

### Phase 1 — Beta ouverte
- Open opt-in contribution
- 500-session threshold for first real fine-tune (base model, corpus total)
- Together.ai Fine-tuning API (transitoire — migration vers Axolotl auto-hébergé en Phase 2)
- Public transparency page

### Phase 2 — Stabilisation
- Automated fine-tuning pipeline (1k–10k sessions)
- Canary deployment: 10% → 50% → 100%
- Auto GPU scaling via RunPod API

### Phase 3 — Maturité
- LoRA adapters axe univers GROG (legrog.org/themes) — seuil 500 sessions/genre
- LoRA adapters axe situation (romance, combat, enquête, diplomatie, exploration, introspection) — seuil 500 sessions/situation
- Pre-merge offline des deux axes dès que les deux adapters individuels existent
- Fallback : situation > univers > base
- Export GGUF pour inférence locale optionnelle
- Multilingual support (FR/EN)
