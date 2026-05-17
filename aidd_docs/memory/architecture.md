---
name: architecture
description: Hub d'architecture — index compact des quatre docs théoriques et des trois docs externes structurants
---

# Architecture — Hub

Ce document est un **point d'entrée**. Il ne contient aucune décision architecturale propre : il oriente vers les documents qui les portent.

## Identité du projet

**Muses** est un service unique, mutualisé entre les instances Suddenly du Fediverse, qui aide les joueurs à écrire de la fiction RP. Architecture : **tirages dans des tables curées + ML léger pour rendre les tirages contextuels**. Pas de LLM génératif, pas de GPU, pas de versions de modèle.

Identité complète et anti-features dans `philosophy.md`.

## Les sept documents qui définissent l'architecture

| Doc | Emplacement | Couvre |
|---|---|---|
| `philosophy.md` | `memory/` | Identité, 8 principes directeurs, anti-features, conventions de nommage |
| `architecture-tables-ml.md` | `memory/` | Pipeline 4-étages (sélecteur / pondérateur / recombinateur / filtreur), 4 niveaux de tables (entités / templates / beats / fragments), asymétrie génération-analyse, carte de couverture |
| `style-coaching.md` | `memory/` | Profil de style auteur, modes confort / challenge, 5 signaux UI, méta-suggestions |
| `learning-and-trust.md` | `memory/` | Bootstrap → continu, online learning des étages, trust contextuel (Beta reputation), réputation d'instance, garde-fous décentralisés, quality gating |
| `axes-and-tags.md` | `memory/external/` | Cinq axes canoniques (univers, situation, rapport_initial, voix, emotion_dominante) avec leur taxonomie de valeurs |
| `data-format.md` | `memory/external/` | Schéma JSONL des rows, conteneur tables, index SQLite, embeddings npy, validation d'ingestion |
| `use-cases.md` | `memory/external/` | Cas d'usage fonctionnels (9 features Issues #72-#89), mapping feature → étages |

## Lecture par rôle

- **Onboarding nouveau contributeur** : `philosophy.md` puis ce hub, puis `architecture-tables-ml.md`.
- **Implémenter un étage du pipeline** : `architecture-tables-ml.md` + `learning-and-trust.md` §3 (online learning).
- **Ajouter une feature** : `use-cases.md` + section concernée d'`architecture-tables-ml.md`.
- **Travailler sur le tagging ou les tables** : `axes-and-tags.md` + `data-format.md`.
- **Sécurité / fédération / trust** : `learning-and-trust.md` §§4-6 + `philosophy.md` §5.
- **UI côté instance Suddenly** : `style-coaching.md` §3 (signaux) + `use-cases.md` §1-2.

## Documents satellites

| Doc | Statut | Couvre / couvrira |
|---|---|---|
| `technical-plan.md` | écrit | Plan d'implémentation : chemin critique + tableau de tâches |
| `analysis-pipeline.md` | projeté | Spec opérationnelle du pipeline d'analyse (projection inversée) |
| `infrastructure.md` | projeté | HA, auth ActivityPub côté service, mode dégradé, déploiement |
| `mining-pipeline.md` | projeté | Pipeline d'extraction des rows depuis le corpus bootstrap |
| Refonte tarifaire | projeté | Économie d'unités d'usage (remplace la grille `use-cases.md` §1.2) |

## Hors périmètre de ce hub

- Toute spécification technique : voir les sept documents ci-dessus.
- Toute roadmap d'implémentation : voir `technical-plan.md`.
- Toute trace historique : voir `lessons_sessions_1-2.md` et `external/issues-analysis.md`.
