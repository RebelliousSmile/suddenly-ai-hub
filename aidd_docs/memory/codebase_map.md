---
name: codebase-structure
description: Project structure documentation
---

# Codebase Structure

```
suddenly-ai-hub/
├── apps/
│   └── gateway/          # FastAPI API Gateway (déployé sur Railway)
│       ├── main.py           # uvicorn entry point (gateway.main:app dans le container)
│       ├── adapter_router.py # routing LoRA adapters
│       ├── auth.py           # HTTP signature auth
│       ├── vllm_client.py    # client vers backend vLLM
│       ├── requirements.txt  # deps Docker (utilisées par Railway)
│       └── Dockerfile        # image Railway (root dir Railway = apps/gateway)
├── pipelines/            # Tout le code Python non-gateway
│   ├── anonymization/    # ex-pipeline/ — anonymize, evaluate, format_corpus, generate_eval
│   ├── crawl_rpv/        # ex-scripts/crawl_rpv/ — scraping + NLLB + scoring
│   ├── evaluation/       # ex-evaluation/ — providers Together/Fireworks, evaluate_lora
│   └── training/         # ex-training/ — Axolotl configs (suddenly-7b/13b.yml, lora-*.yml)
├── scripts/              # Scripts hors pipelines : baseline, infer, evaluate, scrape_*
├── tests/                # pytest, imports via pipelines.*
├── infra/                # docker-compose.yml, mock-instance
├── config/               # scraping_config.ini
├── data/                 # gitignored sauf data/bench/ et fichiers nommés .jsonl tracés
├── aidd_docs/            # toute la doc (memory/, memory/external/, memory/internal/)
├── pyproject.toml        # deps unifiées avec extras [gateway, pipelines, scraper, dev]
├── README.md, CLAUDE.md, AGENTS.md
└── init.py, init.sh      # bootstrap scripts
```

## Conventions imports

- Code dans `pipelines/anonymization/` → `from pipelines.anonymization.X import Y`
- Code dans `pipelines/evaluation/` → `from pipelines.evaluation.X import Y`
- Gateway : `from gateway.X import Y` (résolu via `apps/` dans pythonpath)
- Les tests utilisent `pythonpath = [".", "apps"]` (cf. `pyproject.toml [tool.pytest.ini_options]`).

## Déploiement

- **Gateway** : Railway, build Docker depuis `apps/gateway/Dockerfile` (Root Directory Railway = `apps/gateway`), deps via `apps/gateway/requirements.txt`.
- **Training** : RunPod A100-40G, Axolotl, configs dans `pipelines/training/suddenly-{7b,13b}.yml`.
- **Inference** : vLLM (backend), client dans `gateway/vllm_client.py`.

## Data

- `data/bench/` : corpus de benchmark CC, **in-repo** (petit volume, référencé par `pipelines/crawl_rpv/extract_cyberpunk_samples.py`).
- Reste de `data/` : gitignored (datasets lourds, pas de backup distant — voir mémoire `project_suddenly_ai_hub.md`).
