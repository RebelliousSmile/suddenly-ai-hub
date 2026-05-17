# Changelog

Format inspiré de [Keep a Changelog](https://keepachangelog.com/fr/1.1.0/). Versionnement sémantique.

## [Unreleased]

### Added (pré-MVP, branche `claude/ml-table-selection-algorithm-WdSdd`)

- Service Muses complet : schémas Pydantic des rows, I/O JSONL append-only, index SQLite FTS5, cache embeddings `.npy`, pipeline d'ingestion avec validation et stub signature.
- Pipeline 4 étages CPU-only : sélecteur (tag matching + fallback hiérarchique), pondérateur cosinus, recombinateur non-génératif, filtreur best-of-N.
- API HTTP FastAPI : endpoints `/v1/suggest/{dialogue,action,description,thought,video_prompt}`, `/v1/analyze/{consistency_scene,consistency_session,summary,federated_links}`, `/v1/feedback/signal`, `/v1/admin/coverage`, `/v1/health`.
- `MusesClient` Python avec mode dégradé `MusesUnavailable` sur timeout / 5xx.
- Boucle de feedback : event log JSONL, trust contextuel Beta reputation par (user, axis, value), profil de style auteur, online learning v0, mode challenge avec malus sur rows familières, anti-sleeper, méta-suggestions, snapshots/rollback.
- Pipeline de mining bootstrap : adapter d'anonymisation (regex fallback ou spaCy), extracteur d'entités lexicon-based, classifieur de beats heuristique, script `bootstrap_initial_cell.py` qui peuple la cellule prioritaire (`medieval_fantastique × combat × hostile × solennel × colere`) avec 54 fragments + 6 beats + 11 entités.
- Documentation théorique complète : `philosophy.md`, `architecture.md`, `architecture-tables-ml.md`, `style-coaching.md`, `learning-and-trust.md`, `technical-plan.md`, `infrastructure.md`, `DECISIONS.md`, `LESSONS.md`, `external/axes-and-tags.md`, `external/data-format.md`, `external/use-cases.md`.

### Changed

- **Pivot architectural** : abandon de l'approche LoRA fine-tune au profit de tables curées + ML léger CPU-only. Cf. `DECISIONS.md` D01.
- Cinq axes contextuels canoniques (`univers`, `situation`, `rapport_initial`, `voix`, `emotion_dominante`) — extension du triplet initial pour capturer rapport et émotion comme dimensions distinctes.
- Identifiants normalisés en ASCII snake_case sans accent (`medieval_fantastique`, `narquois`, `colere`…).

### Removed

- Stack pré-pivot : gateway FastAPI Railway, RunPod GPU, Cloudflare R2, providers Together.ai/Fireworks.ai, scripts d'entraînement Axolotl, tests d'évaluation LoRA.
- Docs mémoire LoRA-era purgées (11 fichiers dans `aidd_docs/memory/external/`).

### Security

- Authentification ActivityPub : stub de parsing pour le MVP, vérification cryptographique RSA-SHA256 documentée comme spec M4 dans `infrastructure.md`.
- Aucune dépendance à des providers d'inférence commerciale.
