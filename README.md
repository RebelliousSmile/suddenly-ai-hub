# Suddenly Muses

> Service mutualisé d'assistance créative pour le Fediverse Suddenly. Une muse au sens classique : elle provoque, elle ne sert pas.

[![tests](https://github.com/RebelliousSmile/suddenly-muses/actions/workflows/tests.yml/badge.svg)](.github/workflows/tests.yml)

**Statut** : pré-MVP. Architecture complète (209 tests verts), pas encore déployée en production.

## Identité

**Muses** aide les joueurs des instances Suddenly à écrire de la fiction RP via deux pipelines :

- **Génération** — `suggest_dialogue`, `suggest_action`, `suggest_description`, `suggest_thought`, `video_prompt`. Sortie par tirage et recomposition de lignes de tables curées (jamais de génération token par token).
- **Analyse** — `consistency_scene`, `consistency_session`, `summary`, `federated_links`. Projection inversée du contenu utilisateur sur des tables de patterns.

Mécanique : **tirages dans des tables curées + pipeline ML léger CPU-only**. Détails dans `aidd_docs/memory/architecture.md` (hub).

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
                                                       ├── tables/    JSONL versionnés git
                                                       ├── feedback/  SQLite trust + profil + learner + event log
                                                       ├── snapshots/ copies horodatées (rollback)
                                                       │
                                                       ├── pipeline génération (4 étages)
                                                       │   sélecteur → pondérateur → recombinateur → filtreur
                                                       └── pipeline analyse (projection inversée)
                                                           embedder → matcher → agrégateur
```

Cinq axes contextuels canoniques (cf. `aidd_docs/memory/external/axes-and-tags.md`) :
`univers`, `situation`, `rapport_initial`, `voix`, `emotion_dominante`.

Cinq signaux UI (cf. `aidd_docs/memory/style-coaching.md` §3) : `accept`, `accept_edited`, `reject_off`, `reject_challenge_appreciated`, `ignore`. Le `reject_challenge_appreciated` est ce qui empêche le mode challenge de collapser.

## Démarrage rapide

### Prérequis

- Python ≥ 3.11
- Git

### Install

```bash
git clone https://github.com/RebelliousSmile/suddenly-muses.git
cd suddenly-muses
python -m venv venv
source venv/bin/activate
pip install -e .[api,embeddings]
```

Extras disponibles :

| Extra | Contenu | Quand |
|---|---|---|
| `[api]` | FastAPI, Uvicorn, httpx, cryptography | Service HTTP + client + signature ActivityPub |
| `[embeddings]` | sentence-transformers | Embeddings réels (modèle multilingue CPU ~120 MB) |
| `[pipelines]` | spaCy + modèles FR/EN | Mining bootstrap, anonymisation NER |
| `[scraper]` | Playwright, BeautifulSoup | Sourcing optionnel |
| `[dev]` | pytest | Suite de tests |

### Bootstrap d'une cellule initiale

```bash
PYTHONPATH=. python scripts/bootstrap_initial_cell.py
```

Peuple `tables/bootstrap_cell_medfan_combat_hostile_solennel_colere/` avec 54 fragments + 6 beats + 11 entités depuis `data/test-dataset-rp.jsonl`. Cellule contextuelle : `medieval_fantastique × combat × hostile × solennel × colere`. Le script est idempotent — re-run wipe + recrée.

### Lancer le service localement (mode dev)

```bash
cp .env.example .env
# Édite .env si besoin
export $(cat .env | xargs)

uvicorn muses.api.entrypoint:app --host 127.0.0.1 --port 8000
```

Variables minimales (cf. `.env.example` pour la liste complète) :

```bash
MUSES_TABLE_DIR=tables/bootstrap_cell_medfan_combat_hostile_solennel_colere
MUSES_FEEDBACK_DIR=feedback
MUSES_BIND_HOST=127.0.0.1          # public bind exige MUSES_ADMIN_TOKEN
MUSES_SIGNATURE_MODE=stub          # parse seulement ; "strict" en prod
MUSES_ENCODER=stub                 # ou sentence_transformer
```

### Endpoints

| Méthode | Path | Auth | Description |
|---|---|---|---|
| GET | `/v1/health` | aucune | Liveness + statut signature_mode + counts |
| POST | `/v1/suggest/{dialogue,action,description,thought,video_prompt}` | Signature HTTP | Génération |
| POST | `/v1/feedback/signal` | Signature HTTP | Capture des 5 signaux UI |
| POST | `/v1/analyze/{consistency_scene,consistency_session,summary,federated_links}` | Signature HTTP | Analyse |
| GET | `/v1/admin/coverage` | `X-Admin-Token` | Carte de couverture par cellule |

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
        # mode="challenge" pour pousser l'auteur hors de ses habitudes
    )
    for s in result.suggestions:
        print(s.text)
except MusesUnavailable:
    # Service down → griser le bouton IA, ne pas débiter d'unité d'usage
    pass
```

Le client lève `MusesUnavailable` sur timeout / erreur 5xx ; les 4xx propagent comme `httpx.HTTPStatusError`.

## Déploiement

### Railway (recommandé pour MVP)

`railway.toml` à la racine — config build + deploy versionnée. Hobby plan ~5 USD/mois.

1. Lier le repo GitHub à un projet Railway
2. Provisionner un Volume sur `/data`
3. Set les variables d'environnement (cf. `.env.example`, en particulier `MUSES_ADMIN_TOKEN` obligatoire et `MUSES_SIGNATURE_MODE=strict`)
4. Le déploiement démarre automatiquement sur push `main`

Procédure complète : `aidd_docs/memory/infrastructure.md` § Déploiement Railway. Transposable trivialement vers Fly.io, Render ou tout PaaS Heroku-like.

### Bare-metal (Hetzner ou équivalent)

```bash
# 1. Sur la VM
git clone ... && cd suddenly-muses
pip install -e .[api,embeddings]

# 2. Service systemd avec env vars de prod
sudo systemctl enable muses
sudo systemctl start muses

# 3. Reverse proxy nginx + TLS Let's Encrypt
```

Spec opérationnelle : `aidd_docs/memory/infrastructure.md`.

### Sécurité de production

- `MUSES_SIGNATURE_MODE=strict` active la vérification RSA-SHA256 + résolution acteur ActivityPub + anti-replay (5 min par défaut). Sans ça, le service n'authentifie pas réellement les requêtes.
- `MUSES_ADMIN_TOKEN` est **obligatoire** en bind public (`MUSES_BIND_HOST` non-localhost) — `ConfigError` au startup sinon.
- `MUSES_RATE_LIMIT_PER_MINUTE=60` activé par défaut sur Railway.

### Snapshots

```bash
# Snapshot manuel des stores feedback
PYTHONPATH=. python scripts/snapshot_feedback.py

# Cron horaire suggéré
0 * * * * cd /opt/muses && PYTHONPATH=. python scripts/snapshot_feedback.py
```

## Tests

```bash
pip install -e .[dev,api,embeddings]
pytest tests/
```

209 tests + 1 skip légitime (spaCy si extra `[pipelines]` absent). Suite < 10 s sur CPU récent. CI sur Python 3.11 et 3.12 via GitHub Actions (`.github/workflows/tests.yml`).

## Documentation

Tout est dans `aidd_docs/memory/` :

| Doc | Couvre |
|---|---|
| `philosophy.md` | Identité, 8 principes, anti-features |
| `architecture.md` | Hub d'index vers les 7 docs structurants |
| `architecture-tables-ml.md` | Pipeline 4 étages, 4 niveaux de tables, asymétrie génération/analyse |
| `style-coaching.md` | Profil auteur, modes confort/challenge, 5 signaux UI, méta-suggestions |
| `learning-and-trust.md` | Bootstrap → continu, online learning, trust contextuel, garde-fous |
| `technical-plan.md` | Roadmap M0-M5 |
| `infrastructure.md` | Endpoints, auth ActivityPub, déploiement Railway + bare-metal, snapshots |
| `DECISIONS.md` / `LESSONS.md` | Décisions et leçons du pivot |
| `external/axes-and-tags.md` | Taxonomie canonique des 5 axes |
| `external/data-format.md` | Schéma JSONL des rows |
| `external/use-cases.md` | 9 features fonctionnelles (Issues Suddenly #72-#89) |

## Contribuer

Les contributions sont bienvenues. Convention de commits : Conventional Commits. Le rituel de critique systématique (`0 deal breakers / 0 suggestions`) est attendu pour toute évolution structurante de la mémoire ou de l'architecture.

## License

[MIT](LICENSE)
