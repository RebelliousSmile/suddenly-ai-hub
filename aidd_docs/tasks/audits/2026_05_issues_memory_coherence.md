# Audit — Cohérence issues vs memory
Date: 2026-05-01
Scope: Toutes les issues ouvertes (GitHub) vs memory bank

## Statuts: 5 incohérences trouvées, 5 corrigées
Confidence: 97%

## ✅ Issues conformes (sans correction)
- #6 Setup mono-repo
- #7 Instance Docker mock
- #8 SPIKE modèle de base
- #9 Format données d'entraînement
- #10 Pipeline anonymisation NER
- #11 Config Axolotl
- #12 Fine-tuning initial corpus public
- #13 Jeu de test réservé
- #14 API Gateway ✅ (déjà mis à jour)
- #15 Middleware auth ActivityPub
- #16 Stockage corpus PostgreSQL + S3
- #17 Déploiement beta RunPod
- #18 Discovery instances (PUSH model)
- #22 Déploiement progressif canary
- #23 Page transparence
- #24 Auth ActivityPub conditions réelles
- #26 Métriques qualité
- #27 Scaling GPU RunPod
- #28 Migration Together.ai → Axolotl
- #30 Fine-tuning LoRA + pre-merge
- #31 Modèle multilingue
- #32 Export GGUF inférence locale

## ❌ Incohérences corrigées

### #29 — Milestone Phase 3 → Phase 0
- **Problème** : Le spike de validation LoRA était en Phase 3, alors qu'il conditionne les configs Axolotl de Phase 0 (#11).
- **Correction** : Déplacé en Phase 0 — Fondations.

### #20 — Client LLM sans genre/situation
- **Problème** : L'implémentation du client ne mentionnait pas les paramètres `genre` et `situation` décidés pour chaque requête.
- **Correction** : Tâches ajoutées pour passer et gérer ces paramètres.

### #33 — Tâche "définir format API" déjà résolue
- **Problème** : La tâche demandait de définir le format de `genre`/`situation` dans l'API, déjà décidé dans #14 et `architecture.md`.
- **Correction** : Tâche remplacée par une vérification de cohérence avec #14.

### #21 — Ambiguïté seuil 500 (total vs par catégorie)
- **Problème** : "500 sessions" sans préciser que c'est le total corpus (base model), distinct du 500/catégorie des LoRA (Phase 3).
- **Correction** : Clarification ajoutée dans le body.

### #25 — Même ambiguïté seuil 500
- **Problème** : Fine-tuning automatisé Phase 2 = modèle de base (corpus total), pas les LoRA/catégorie.
- **Correction** : Clarification ajoutée.

## Incohérences restantes dans la memory

### project_brief.md — 2 occurrences "OpenAI-compatible" obsolètes
- Ligne 17 : "OpenAI-compatible REST API" → à corriger (contrainte abandonnée)
- Ligne 44 : `POST /v1/chat/completions` — "OpenAI-compatible inference" → à corriger

### project_brief.md — Model Routing by Feature non intégré aux LoRA
- La table `suggest_short → suddenly-7b-q4` etc. ne précise pas que `genre`/`situation` s'ajoutent à ce routing.
- Ces deux dimensions sont complémentaires, pas concurrentes.
