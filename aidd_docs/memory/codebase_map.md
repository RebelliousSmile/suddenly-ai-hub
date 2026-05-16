---
name: codebase-structure
description: Project structure documentation
---

# Codebase Structure

```mermaid
flowchart TD
---
title: suddenly-ai-hub — Macro Overview
---
    Gateway["gateway/ (API Gateway)"]
    Pipeline["pipeline/ (Ingestion & Format)"]
    Training["training/ (Fine-tuning)"]
    Infra["infra/ (Docker & RunPod)"]
    Docs["docs/ (Architecture & Transparency)"]
    Storage["Storage"]
    Inference["Inference"]
    Model["Model"]

    Gateway -- "routes requests" --> Inference
    Pipeline -- "feeds" --> Storage
    Training -- "produces" --> Model
    Infra -- "deploys" --> Gateway
    Infra -- "deploys" --> Pipeline
    Infra -- "deploys" --> Training
    Docs -.-> Gateway
    Docs -.-> Pipeline
    Docs -.-> Training
```

## Current state
- Phase 0: only AIDD tooling exists (`.claude/`, `aidd_docs/`, `.aidd/`)
- No application code yet — `gateway/`, `pipeline/`, `training/`, `infra/` not created

## Planned modules
- `gateway/` — FastAPI API Gateway
- `pipeline/` — ingestion, anonymisation, formatage
- `training/` — fine-tuning scripts (Axolotl configs)
- `infra/` — Docker Compose, RunPod scripts
- `docs/` — architecture.md, data-format.md, transparency.md
