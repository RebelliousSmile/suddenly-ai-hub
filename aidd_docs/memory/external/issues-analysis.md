# Analyse des Issues — suddenly-ai-hub

**2026-05-15** — Dernière mise à jour — **20 issues fermées, 0 ouvertes**

---

## ✅ Toutes les issues sont fermées ou mises en attente

### Issues fermées avec validation partielle

| # | Issue | Résultat | Statut |
|---|-------|----------|--------|
| #44 | Préparer l'environnement Python | venv-together avec SDK + pip installées | ✅ Terminé |
| #45 | Tester l'API Together.ai | Upload + FT job complété, benchmark pivot Fireworks | ✅ Terminé |
| #46 | Dataset test minimal | 10 exemples validés, uploadé, utilisé dans FT | ✅ Terminé |
| #47 | Scrape La Cour d'Obéron | Mis de côté (forum en refonte, pas accessible) | ⏸️ En attente |
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
| #29 | [SPIKE] Archi LoRA 2 axes + pre-merge | Architecture validée et documentée | `lora-strategy.md` |
| #33 | [SPIKE] Taxonomie genres/situations | Taxonomie validée et documentée | `taxonomia.md` |

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

## 📚 Documentation créée (aidd_docs/memory/external/)

| Fichier | Contenu |
|---------|---------|
| `lora-strategy.md` | Architecture LoRA 2 axes, fusion offline, fallback hierarchy, seuils |
| `taxonomy.md` | 14 genres + 6 situations, mécanisme de tagging |
| `benchmark-fireworks-vs-together.md` | Comparatif prix/inférence/serverless, recommandation hybride |

---

## 📋 État des dépendances restantes

### Pipeline de données (en attente de corpus réel)
```
#47 (Scrape) → #48 (Nettoyage) → #49 (Conversion) → #51 (Fine-tuning)
  ⏸️            ⏸️              ⏸️              ⏸️
```

**Blocage** : Aucun corpus réel disponible. Options :
- Corpus synthétique généré par LLM
- Forums RP francophones publics
- Logs Discord/IRC publics

**Options corpus identifiées** :
- **#58 Ren'Py** → Visual novels Ren'Py publics GitHub (2M tokens est.)
- **#59 Playwright** → Scraping sites modernes JS (completé, lourd)
- **#60 Google Books** → Livres domaine public FR (20M tokens est., style uniquement)

### Modèle de base (en attente de GPU)
- Configurations Axolotl : ✅ prêtes
- Dataset test : ✅ validé
- GPU disponibles : ❌ (Together/Fireworks épuisés)
- Solution envisagée : RunPod A100-40G

### LoRA (en attente de modèle de base FT)
- Architecture : ✅ documentée (`lora-strategy.md`)
- Taxonomie : ✅ documentée (`taxonomy.md`)
- Entraînement : ⏸️ en attente modèle de base

### Production/Features (hors chemin critique)
- Toutes les features (hub beta, discovery, rollout, etc.) en attente du modèle FT opérationnel
- Pas de dépendance entre elles — à lancer séquentiellement après FT
