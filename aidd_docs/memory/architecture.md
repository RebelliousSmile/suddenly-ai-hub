---
name: architecture
description: Module architecture and structure
---

# Architecture

## Language/Framework

- FastAPI — API Gateway (Python)
- vLLM — Inference engine (GPU, RunPod)
- Axolotl — Fine-tuning scripts (configs)
- PostgreSQL — Hub database
- S3-compatible — Object storage (Backblaze B2 or Hetzner)
- ActivityPub — Authentication protocol

```mermaid
flowchart LR
---
title: Hub Architecture
---
    Instances["Fediverse Instances"]
    Gateway["API Gateway (FastAPI)"]
    Inference["Inference Engine (vLLM)"]
    Ingestion["Data Ingestion Pipeline"]
    Training["Training Pipeline (Axolotl)"]
    Postgres["PostgreSQL"]
    S3["S3 Storage"]

    Instances -- inference request --> Gateway
    Instances -- contribution --> Gateway
    Gateway --> Inference
    Gateway --> Ingestion
    Ingestion --> Postgres
    Ingestion --> S3
    Ingestion -.-> Training
    Training -.-> Inference
```

## API Endpoints

- `POST /v1/chat/completions` — inférence (chemin conservé par convention, pas par contrainte OpenAI)
- `GET /v1/models` — modèles et adapters disponibles
- `GET /v1/health` — statut du hub
- `POST /v1/contribute` — soumission de session (**push** depuis instance après opt-in, modèle retenu définitivement)
- `GET /v1/stats` — statistiques publiques

> Pas de contrainte de compatibilité OpenAI SDK — le client est écrit par nous (#20).

## Models

### Base models (full fine-tuned on general RP corpus)
- `suddenly-7b` — 7B
- `suddenly-7b-q4` — 7B quantized
- `suddenly-13b` — 13B

### LoRA adapters — deux axes indépendants, pre-merged au service

vLLM ne stacke pas deux LoRA en live. Solution : entraîner chaque axe séparément, pre-merger les delta weights offline, servir un seul adapter pre-merged par requête.

**Axe univers** (taxonomie GROG — legrog.org/themes, par priorité corpus) :
- `lora-medieval-fantastique`, `lora-historique-fantastique`, `lora-scifi`
- `lora-contemporain-fantastique`, `lora-space-opera`, `lora-contemporain`
- `lora-post-apocalyptique`, `lora-cyberpunk`, `lora-super-heros`, `lora-oriental-manga`
- `lora-generique` — sessions sans genre identifié ou corpus "Générique"/"Inclassables" GROG

**Axe situation** :
- `lora-romance`, `lora-combat`, `lora-enquete`
- `lora-diplomatie`, `lora-exploration`, `lora-introspection`

### Déclenchement et pre-merge
- Chaque adapter entraîné dès que son corpus propre atteint **500 sessions**
- Pre-merge `lora-{univers}-{situation}` dès que les **deux adapters individuels existent**
- Pre-merge déclenché automatiquement par le pipeline, pas d'attente de corpus combiné

### Format des paramètres
```json
POST /v1/chat/completions
{
  "model": "suddenly-7b",
  "messages": [...],
  "genre": "medieval-fantastique",
  "situation": "combat"
}
```
- `genre` et `situation` : optionnels, slugs minuscules avec tirets
- Enum dynamique : chargé depuis la DB au démarrage (= adapters réellement déployés)
- Valeur inconnue → HTTP 422 avec liste des valeurs acceptées
- Enum défini dans `docs/taxonomy.md` (spike #33)

### Fallback complet (par ordre de priorité)
1. `lora-{univers}-{situation}` — les deux fournis et pre-merged disponible
2. `lora-{situation}` — situation fournie (univers absent ou pre-merge inexistant)
3. `lora-{univers}` — univers fourni, situation absente
4. `lora-generique` — ni genre ni situation fournis, adapter entraîné
5. `suddenly-7b` / `suddenly-13b` base — aucun adapter disponible

## Authentication

- Instance sends `Authorization: ActivityPub {instance_domain}` + HTTP signature
- Hub fetches instance actor public key from instance
- Hub verifies HTTP signature against public key

## Storage

- PostgreSQL — corpus metadata, session records
- S3-compatible — corpus files (Backblaze B2 or Hetzner)

## Training Data Format

- Chat messages JSON — `system`, `user`, `assistant` roles

## Planned Structure

```
gateway/     # FastAPI API Gateway
pipeline/    # Ingestion, anonymisation, formatting
training/    # Fine-tuning scripts (Axolotl configs)
infra/       # Docker Compose, RunPod scripts
docs/        # architecture, data-format, transparency
```

## Naming Conventions

- Not yet established — no code written
