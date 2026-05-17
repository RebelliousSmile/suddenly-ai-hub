# Analyse des Issues — suddenly-ai-hub

**2026-05-15** — Dernière mise à jour — **20 issues fermées, 6 ouvertes**

> **Statut : snapshot pré-pivot.** Ce document reflète l'état du projet à la mi-mai 2026, avant l'abandon de l'approche LoRA / fine-tune (cf. `philosophy.md` § Contexte du pivot dans `architecture-tables-ml.md`). Conservé comme trace historique. Pour l'état courant, voir les issues GitHub et les quatre documents théoriques (`philosophy.md`, `architecture-tables-ml.md`, `style-coaching.md`, `learning-and-trust.md`).

---

## ✅ Toutes les issues sont fermées ou mises en attente

### Issues fermées avec validation partielle

| # | Issue | Résultat | Statut |
|---|-------|----------|--------|
| #44 | Préparer l'environnement Python | venv-together avec SDK + pip installées | ✅ Terminé |
| #45 | Tester l'API Together.ai | Upload + FT job complété, benchmark pivot Fireworks | ✅ Terminé |
| #46 | Dataset test minimal | 10 exemples validés, uploadé, utilisé dans FT | ✅ Terminé |
| #47 | Scrape La Cour d'Obéron | Mis de côté (décision éthique : pas de scraping de forums PB) | ⏸️ Archivé |
| #48 | Nettoyage/anonymisation des données | Pipeline prêt, pas de données à traiter | ⏸️ En attente |
| #49 | Convertir en JSONL Axolotl | Format validé sur dataset test, pas de corpus réel | ⏸️ En attente |
| #50 | Configurer Axolotl | 4 configs créées (suddenly-7b, 13b, lora-situation, lora-univers) | ✅ Terminé |
| #51 | Lancer le premier fine-tuning | Pipeline validé mais GPU indisponible | ⏸️ En attente |
| #52 | Tester et évaluer le modèle | Grille d'évaluation définie, pas de modèle à évaluer | ⏸️ En attente |
| #53 | Itérer et améliorer | Activité continue, pas de ticket dédié | ⏸️ En attente |
| #54 | Documenter | Documentation existante couvre les aspects nécessaires | ⏸️ En attente |

### Issues fermées avec décision documentée (SPIKEs)

| # | Issue | Résultat | Biblio |
|---|-------|----------|--------|
| #29 | [SPIKE] Archi LoRA 2 axes + pre-merge | Architecture LoRA validée à l'époque — depuis abandonnée | obsolète, voir `architecture-tables-ml.md` |
| #33 | [SPIKE] Taxonomie genres/situations | Taxonomie 2 axes validée à l'époque — depuis étendue à 5 axes canoniques | voir `axes-and-tags.md` |

### Issues fermées en attente (production/features)

Ces issues définissent des features futures à lancer **après** que le pipeline de données et le modèle de base sont opérationnels.

| # | Issue | Dépendance principale |
|---|-------|----------------------|
| #16 | Stockage corpus PostgreSQL+S3 | Pipeline de données opérationnel |
| #17 | Déploiement hub beta | Modèle de base FT opérationnel |
| #18 | Discovery instances | Hub beta déployé |
| #20 | Client LLM côté instance | Modèle FT opérationnel |
| #21 | Premier FT sur données Suddenly | #47, #48, #49 |
| #22 | Rollout progressif | Hub beta stable |
| #23 | Page transparence financière | Feature communication |
| #24 | Auth ActivityPub réelle | #17, #18 |
| #25 | FT automatisé déclenché par seuil | #47, #51 |
| #26 | Métriques qualité automatisées | #51, #52 |
| #27 | Scaling GPU auto | Feature ops |
| #28 | Migration auto-hébergé | Optimisation future |
| #30 | Fine-tuning LoRA axe univers + situation | Modèle de base FT opérationnel |
| #31 | Modèle multilingue FR/EN | Feature avancée, pas maintenant |
| #32 | Export modèle pour inférence locale | Feature optionnelle |

---

## 📚 Documentation créée (snapshot)

| Fichier | Contenu | Statut courant |
|---------|---------|----------------|
| `lora-strategy.md` | Architecture LoRA 2 axes, fusion offline, fallback hierarchy, seuils | Supprimé — remplacé par `architecture-tables-ml.md` |
| `taxonomy.md` | 14 genres + 6 situations, mécanisme de tagging | Renommé en `axes-and-tags.md`, étendu à 5 axes canoniques |
| `benchmark-fireworks-vs-together.md` | Comparatif prix/inférence/serverless | Toujours dans `memory/`, mais sans pertinence opérationnelle depuis le pivot |
| `sources-litteraires.md` | Guide juridique & sources légales FR | Jamais créé |

---

## 📋 État des dépendances restantes

### Pipeline de données (re-direction stratégique)

**Décision 2026-05-15** : Arrêt du scraping de forums Pavé de France (PB). Raison éthique — les forums de création privée ne sont pas des sources ouvertes. Direction actuelle : **sources littéraires + Visual Novels publics**.

```
#62 (Horreur — pipeline pilote) → #63 (Genres — industrialisation) → #64 (Cyberpunk — traduction)
   ⏸️                              ⏸️                                  ⏸️
```

**Sources prioritaires actuelles** :

| Source | Type | Volume est. | Statut juridique |
|--------|------|-------------|------------------|
| **Visual Novels Ren'Py** (#58) | Dialogues VN publics GitHub | ~2M tokens | Publics (GitHub releases) |
| **Classiques FR** (Wikisource/Gallica) | Romans domaine public | ~20M tokens | DP France (†<1955) |
| **Traductions CC** (#64) | Romans EN → FR (Doctorow, Watts) | ~2M tokens | CC BY-NC-SA (traduit) |

**Sources archivées / non prioritaires** :

| Source | Raison de la mise de côté |
|--------|--------------------------|
| **#47 La Cour d'Obéron** | Décision éthique : pas de scraping de forums PB |
| **#59 Playwright** | Trop lourd, scraping de sites modernes JS |
| **#60 Google Books** | Déplacé en source secondaire (envisagé si besoin) |

### Modèle de base (en attente de GPU)
- Configurations Axolotl : ✅ prêtes
- Dataset test : ✅ validé
- GPU disponibles : ❌ (Together/Fireworks épuisés)
- Solution envisagée : RunPod A100-40G

### LoRA (entièrement annulé suite au pivot)
- Architecture LoRA : ❌ abandonnée — voir `architecture-tables-ml.md` pour la nouvelle archi tables + ML
- Taxonomie 2 axes : étendue à 5 axes canoniques dans `philosophy.md` § Conventions
- Entraînement : ❌ ne s'applique plus, plus de fine-tune batch

### Production/Features (hors chemin critique)
- Toutes les features (hub beta, discovery, rollout, etc.) en attente du modèle FT opérationnel
- Pas de dépendance entre elles — à lancer séquentiellement après FT

---

## 🆕 Issues ouvertes

| # | Issue | Description |
|---|-------|-------------|
| #57 | Tests de validation LoRA par genre/situation | Script Python + prompts, scoring 1-5, rapport JSON/CSV |
| #58 | Scrapper Visual Novels Ren'Py depuis GitHub | Collecte dialogues VN publics (~2M tokens) |
| #59 | Scraping JavaScript avec Playwright | Sites modernes JS (mis de côté, trop lourd) |
| #62 | Phase 1 : Pipeline pilote Horreur | Téléchargement ePub → nettoyage → classification Ollama → chunking → audit |
| #63 | Phase 2 : Industrialisation corpus genres | Répéter pipeline Horreur pour SF, Fantasy, Polar, Steampunk, Aventure |
| #64 | Phase 3 : LoRA Cyberpunk via traduction | Workflow curation EN → traduction Mistral/Ollama → export JSONL |
| #65 | Répartition provider Together.ai vs Fireworks.ai | Stratégie hybride : Together inférence, Fireworks batch/FT/embeddings |
