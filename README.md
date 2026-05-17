# Suddenly Muses

> Service mutualisé d'assistance créative pour le Fediverse Suddenly. Une muse au sens classique : elle provoque, elle ne sert pas.

## Identité

**Muses** aide les joueurs des instances Suddenly à écrire de la fiction RP en proposant des suggestions (dialogue, action, description, pensée intérieure, prompt vidéo) et des analyses (cohérence de scène/session, résumé, suggestions de liens fédérés).

Sa mécanique interne — **tirages dans des tables curées + pipeline ML léger CPU-only** — est documentée dans `aidd_docs/memory/architecture.md` (hub) et les sept docs structurants qu'il pointe.

## Ce que Muses n'est pas

- Pas un LLM, pas un chatbot.
- Pas un substitut à l'auteur — il propose, l'auteur arbitre.
- Pas un système à inférence payante par token. Aucune génération autoregressive.
- Pas une instance Suddenly de plus — c'est un service unique mutualisé.

Détail : `aidd_docs/memory/philosophy.md` §8.

## Architecture en une page

```
[ Instance Suddenly ] ──HTTPS signé ActivityPub──► [ Service Muses CPU-only ]
                                                       │
                                                       ├── tables/   (JSONL versionnés)
                                                       ├── feedback/ (SQLite trust + profil + learner)
                                                       └── pipeline 4 étages
                                                            sélecteur → pondérateur → recombinateur → filtreur
```

Cinq axes contextuels canoniques (cf. `aidd_docs/memory/external/axes-and-tags.md`) :
`univers`, `situation`, `rapport_initial`, `voix`, `emotion_dominante`.

## Démarrage rapide

### Install

```bash
git clone https://github.com/RebelliousSmile/suddenly-muses.git
cd suddenly-muses
python -m venv venv
source venv/bin/activate
pip install -e .[api,embeddings]
```

Extras disponibles :
- `[api]` — FastAPI, Uvicorn, httpx (service HTTP + client)
- `[embeddings]` — sentence-transformers (modèle multilingue CPU pour embeddings réels)
- `[pipelines]` — spaCy + modèles FR/EN (mining bootstrap, anonymisation NER)
- `[scraper]` — Playwright, BeautifulSoup (sourcing optionnel)
- `[dev]` — pytest

### Bootstrap d'une cellule initiale

```bash
PYTHONPATH=. python scripts/bootstrap_initial_cell.py
```

Peuple `tables/bootstrap_cell_medfan_combat_hostile_solennel_colere/` avec ~54 fragments + 6 beats + 11 entités depuis `data/test-dataset-rp.jsonl`. Cellule contextuelle : `medieval_fantastique × combat × hostile × solennel × colere`.

### Lancer le service localement

```bash
export MUSES_TABLE_DIR=tables/bootstrap_cell_medfan_combat_hostile_solennel_colere
export MUSES_FEEDBACK_DIR=feedback
export MUSES_ADMIN_TOKEN=local-dev-only

uvicorn muses.api.entrypoint:app --host 127.0.0.1 --port 8000
```

Le service écoute :
- `GET  /v1/health`
- `POST /v1/suggest/{dialogue,action,description,thought,video_prompt}`
- `POST /v1/feedback/signal`
- `POST /v1/analyze/{consistency_scene,consistency_session,summary,federated_links}`
- `GET  /v1/admin/coverage` (header `X-Admin-Token`)

Spec opérationnelle complète : `aidd_docs/memory/infrastructure.md`.

### Client Python (côté instance Suddenly)

```python
from muses.client import MusesClient, MusesUnavailable
from muses.schemas.tags import AxialTags

client = MusesClient(base_url="https://muses.suddenly.social")

try:
    result = client.suggest(
        feature="dialogue",
        context_text="Le chevalier brandit son épée face au voleur.",
        context_tags=AxialTags(
            univers=["medieval_fantastique"],
            situation=["combat"],
            rapport_initial=["hostile"],
            voix=["solennel"],
            emotion_dominante=["colere"],
        ),
        signature='keyId="https://exemple.tld/users/alice#main-key",signature="..."',
    )
    for s in result.suggestions:
        print(s.text)
except MusesUnavailable:
    # Service down → griser le bouton IA, ne pas débiter d'unité d'usage
    pass
```

## Tests

```bash
pip install -e .[dev,api,embeddings]
pytest tests/
```

## Documentation

Tout est dans `aidd_docs/memory/` :

- `philosophy.md` — identité, 8 principes, anti-features
- `architecture.md` — hub d'index vers les 7 docs structurants
- `architecture-tables-ml.md` — pipeline 4 étages
- `style-coaching.md` — 5 signaux UI, modes confort/challenge
- `learning-and-trust.md` — online learning, trust contextuel, garde-fous
- `technical-plan.md` — roadmap M0-M5
- `DECISIONS.md` / `LESSONS.md` — décisions et leçons du pivot
- `infrastructure.md` — endpoints, auth, déploiement, snapshots
- `external/axes-and-tags.md` — taxonomie canonique des 5 axes
- `external/data-format.md` — schéma JSONL des rows
- `external/use-cases.md` — 9 features fonctionnelles (Issues Suddenly #72-#89)

## License

MIT
