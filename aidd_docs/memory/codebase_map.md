---
name: codebase-structure
description: Carte de la structure de code du repo suddenly-muses (état courant)
---

# Codebase map

État courant du repo après le pivot. Les artefacts LoRA-era (`apps/gateway/`, `pipelines/training/`, `pipelines/evaluation/`) ont été supprimés dans la série de commits précédant le pivot architectural.

```
suddenly-muses/
├── apps/
│   └── playground/         # GUI Gradio pour exploration / banc d'essai (hérité, à reconsidérer)
├── pipelines/              # Code Python — outils non-API
│   ├── anonymization/      # Anonymisation NER des noms propres (hérité, à adapter pour produire des rows — T11)
│   └── crawl_rpv/          # Scraping + collecte (hérité, à adapter pour pipeline mining — T12)
├── scripts/                # Scripts utilitaires (scrapers, helpers, learn/changelog)
├── tests/                  # pytest — beaucoup de tests obsolètes à purger en T01-T02
├── config/                 # Fichiers de config (scraping_config.ini)
├── data/                   # Données — `data/bench/`, `data/renpy-corpus.jsonl`, `data/test-dataset-rp.jsonl` tracés ; reste gitignored
├── aidd_docs/              # Mémoire et règles AIDD
│   ├── memory/             # Mémoire projet (philosophy, architecture-tables-ml, etc.)
│   │   ├── external/       # Mémoire externe (use-cases, axes-and-tags, data-format, etc.)
│   │   └── internal/       # Mémoire interne (templates, PROJECT.md, MANIFEST)
│   └── ...
├── .claude/                # Conventions aidd-framework : agents, commands, rules, skills
├── pyproject.toml          # Deps Python — extras `[gateway, pipelines, scraper, playground, dev]` ; les extras `gateway` et `playground` survivent à l'ancienne stack et doivent être nettoyés (T03)
├── CLAUDE.md, AGENTS.md    # Context et règles agent
├── README.md               # À réécrire selon `philosophy.md`
└── init.py, init.sh        # Bootstrap dev
```

## Composants à venir (cf. `technical-plan.md`)

Le code du **service Muses** n'existe pas encore. Cibles structurelles à créer :

```
suddenly-muses/
├── muses/                  # (à créer) Code du service Muses
│   ├── tables/             # I/O JSONL + index SQLite + npy (T05-T09)
│   ├── ingestion/          # Validation + signature + insertion (T10)
│   ├── pipeline/           # Les 4 étages (T16-T20)
│   ├── api/                # HTTP endpoints (T21)
│   ├── learning/           # Online learning (T29-T31)
│   └── trust/              # Beta reputation + instance weight (T27-T28)
├── tables/                 # (à créer) Persistance des rows (JSONL versionné)
└── ...
```

## Conventions imports

- `pipelines.anonymization.*`, `pipelines.crawl_rpv.*` — accédés depuis la racine.
- `tests/` utilise `pythonpath = ["."]` (à mettre à jour dans `pyproject.toml [tool.pytest.ini_options]` lors du cleanup T03).
- Le code à venir du service (`muses/*`) suivra la même convention.

## Données dans le repo

- `data/bench/` : corpus de benchmark, **in-repo** (petit volume).
- `data/renpy-corpus.jsonl` : extraits VN Ren'Py (~27 KB), in-repo pour le bootstrap.
- `data/test-dataset-rp.jsonl` : dataset de test, in-repo.
- Le reste de `data/` est gitignored.

**Question ouverte** : emplacement des tables peuplées. Deux options :

1. Répertoire dédié versionné `tables/` à la racine (cohérent avec `data-format.md` qui parle de « JSONL versionné en git »).
2. Sous-dossier `data/tables/` resté gitignored (moins de pollution diff, perd l'audit git).

À trancher en M0 selon le volume attendu des premières tables.

## Déploiement

Pas encore défini. Cf. `deployment.md` pour les pistes envisagées et `technical-plan.md` T23 (déploiement v0) et T37 (production).
