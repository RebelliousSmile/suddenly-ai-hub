---
name: deployment
description: Infrastructure et déploiement — pistes et contraintes
---

# Déploiement

> Document de synthèse. La spec opérationnelle complète (HA, auth, mode dégradé, monitoring) ira dans `infrastructure.md` à venir (cf. `architecture-tables-ml.md` § Hors périmètre et `technical-plan.md` T39).

## Contraintes structurelles

- **CPU-only** — aucun GPU en production (cf. `philosophy.md` §7). Les modèles ML (classifieurs, embeddings, rerankers) sont conçus pour tourner sur CPU.
- **Service unique mutualisé** — une seule instance Muses pour toutes les instances Suddenly du Fediverse. Le SPOF est assumé pour le MVP, sa résilience est traitée plus tard.
- **Tables sur disque** — pas de DB serveur pour le MVP (JSONL versionné en git + index SQLite + embeddings `.npy`, cf. `external/data-format.md`).
- **Auth ActivityPub** — toutes les requêtes entrantes signées par une instance authentifiée.

## Hébergement envisagé (à figer plus tard)

| Phase | Cible | Critère de choix |
|---|---|---|
| Dev / M2 | Local (poste développeur) ou VM CPU modeste (~$5/mois) | Itération rapide, pas de SLA |
| Production / M4 | Single VM CPU avec disque suffisant pour les tables (Hetzner, OVH, ou équivalent EU pour la souveraineté des données) | Coût ≪ providers d'inférence commerciaux, scaling vertical suffisant pour les premières instances connectées |
| Plus tard | À évaluer selon trafic réel | HA actif/passif éventuellement, à traiter dans `infrastructure.md` |

## Stockage des tables

- Fichiers JSONL versionnés en git dans le repo, ou dans un repo dédié séparé selon volume.
- Sauvegardes : git suffit tant que la croissance reste modérée. À reconsidérer si on dépasse plusieurs dizaines de MB.
- L'**index SQLite** et les **embeddings `.npy`** sont **reconstruits** depuis les JSONL — pas la source de vérité, donc gitignorés.

## Stockage des données opérationnelles (proposition pour le MVP)

`learning-and-trust.md` ne fixe pas la techno de persistance. Proposition pour le MVP, à figer dans `infrastructure.md` à venir :

- **Trust + profils de style** : SQLite local au service Muses, sauvegardé périodiquement.
- **Event log** des signaux UI : append-only sur disque (rotation périodique), utilisé pour l'online learning et l'audit.
- **Snapshots des poids ML** : sur disque, horodatés, pour rollback (cf. `learning-and-trust.md` § Snapshots et rollback).

## Ce qui a été retiré dans le pivot

- **Railway + PostgreSQL** (ancien gateway FastAPI) — supprimé.
- **RunPod GPU spot** (entraînement et inférence vLLM) — supprimé.
- **Cloudflare R2** (stockage modèles + corpus) — pas nécessaire pour le MVP, à reconsidérer si le volume de tables ou d'event logs justifie un object store séparé.
- **Together.ai, Fireworks.ai** — plus aucune dépendance à des providers d'inférence commerciaux.

## Flux de données (synthèse)

```
Instance Suddenly --- (HTTP signé ActivityPub) ---> Service Muses
                                                    |
                                                    +-- Tables JSONL (lecture)
                                                    +-- Index SQLite + embeddings npy
                                                    +-- État online (trust, profils, signaux)
                                                    |
                                                    +-- Pipeline 4 étages → réponse
                                                    |
                                                    +-- (Si opt-in) ingestion row → tables
                                                    +-- (Toujours) signal UI → online learning
```

## Hors périmètre de ce document

- Spec opérationnelle de l'API (endpoints, schémas request/response, codes d'erreur) — `infrastructure.md` à venir.
- Plan de continuité, RPO/RTO, HA actif/passif — `infrastructure.md`.
- Coûts détaillés en production — à chiffrer après M4.
- CI/CD — à mettre en place au moment du M2 / M4.
