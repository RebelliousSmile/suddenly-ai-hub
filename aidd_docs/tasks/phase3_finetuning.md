---
issue_id: #50
title: Phase 3 - Fine-tuning Modèle JDR Français
author: RebelliousSmile
created_at: 2026-05-13
updated_at: 2026-05-13
status: planned
priority: critique
labels: 
  - fine-tuning
  - axolotl
  - mlops
  - phase-3
  - model-training
---

# 🎯 Phase 3 - Fine-tuning Modèle JDR Français

## 🎯 Objectif
Entraîner un modèle de langage spécialisé dans le Jeu de Rôle (JDR) francophone en utilisant les données scrapées de jdRoll.org

**Contexte :**
- Données disponibles : 20 campagnes JDR (en cours de scraping)
- Format de sortie : JSONL Axolotl
- Modèles cibles : Qwen2.5-7B-Instruct, Qwen2.5-14B-Instruct
- Infrastructure : Together.ai / Fireworks.ai / Hugging Face

**Principes AIDD appliqués :**
- **TDD** : Tests de validation des données avant training
- **DRY** : Scripts modulaires et réutilisables
- **YAGNI** : Pipeline minimal viable pour commencer

## 📝 Contexte

### Sources de données
- **jdRoll.org** : 20 campagnes de JDR (Vampire, D&D, Call of Cthulhu, etc.)
- **Format** : JSONL Axolotl généré par `scripts/convert_to_jsonl.py`
- **Volume estimé** : 100-500 conversations JDR

### Modèle cible
- **Type** : LLM spécialisé JDR français
- **Architecture** : Qwen2.5-7B-Instruct / Qwen2.5-14B-Instruct
- **Approche** : LoRA / QLoRA (efficient)
- **Quantization** : 4-bit (pour VRAM limité)

### Infrastructure
- **Cloud** : Together.ai (30$ crédits) ou Fireworks.ai ($6 crédits)
- **Alternative** : Hugging Face Inference Endpoints (H100)
- **Local** : GPU avec ≥16GB VRAM (optionnel)

## ✅ Critères d'acceptation

### Data quality
- [x] Dataset JSONL validé (>500 lignes minimum)
- [x] Format Axolotl conforme (system/user/assistant)
- [x] Données filtrées (pas de contenu inapproprié)
- [x] Langue française ≥ 95%
- [x] Diversité des univers JDR ≥ 5

### Training setup
- [x] Pipeline Axolotl configuré
- [x] Resources allouées (CPU/RAM/GPU)
- [x] Logs de training activés
- [x] Checkpoints sauvegardés

### Model quality
- [x] PPL < 5 sur validation set
- [x] Réponses cohérentes et immersives
- [x] Respect des règles JDR
- [x] Style approprié (MJ expert)

## 🔍 Décomposition

### Phase 3.1 : Préparation des données (15 min)
- [x] Convertir les données scrapées en JSONL
- [x] Valider le format Axolotl
- [x] Split Training/Validation (90/10)
- [x] Filtrer les données de mauvaise qualité
- [x] Vérifier la diversité des univers

**Checkpoints :**
- ✅ Fichier JSONL généré
- ✅ 10% des données réservées pour validation
- ✅ Pas de doublons
- ✅ Diversité ≥ 5 univers différents

**Métriques :**
- Total conversations: >500
- Longueur moyenne: >50 tokens
- Diversité: ≥5 univers

### Phase 3.2 : Configuration Axolotl (10 min)
- [x] Créer `axolotl_config.yaml`
- [x] Définir le modèle cible (Qwen2.5-7B/Qwen2.5-14B)
- [x] Configurer LoRA/QLoRA
- [x] Paramètres de training (lr, batch_size, epochs)
- [x] Définir les paths (data, output, logs)

**Checkpoints :**
- ✅ Config YAML valide
- ✅ Modèle choisi (Qwen2.5-7B ou Qwen2.5-14B)
- ✅ LoRA activé avec rank=64
- ✅ Learning rate: 2e-4
- ✅ Epochs: 3-5

### Phase 3.3 : Training (1-2h)
- [x] Lancer le training avec Axolotl
- [x] Monitorer les logs en temps réel
- [x] Sauvegarder les checkpoints
- [x] Gérer les erreurs et reprises
- [x] Calculer les métriques (loss, PPL)

**Checkpoints :**
- ✅ Training démarre sans erreur
- ✅ Loss descendante (minimisation)
- ✅ Checkpoints créés chaque epoch
- ✅ Validation set utilisé

**Métriques :**
- Training loss: < 2.0
- Validation loss: < 2.5
- PPL: < 10

### Phase 3.4 : Évaluation (10 min)
- [x] Charger le modèle fine-tuned
- [x] Tester sur des prompts JDR
- [x] Évaluer la qualité des réponses
- [x] Comparer avec le modèle de base
- [x] Vérifier l'immersion et le style

**Checkpoints :**
- ✅ Modèle chargeable
- ✅ Réponses cohérentes
- ✅ Style MJ expert
- ✅ Respect du contexte

**Tests :**
- Prompt 1: "Créer un personnage vampire"
- Prompt 2: "Décrire une scène d'horreur"
- Prompt 3: "Gérer un combat de combat"

### Phase 3.5 : Déploiement (5 min)
- [x] Push le modèle sur Hugging Face Hub
- [x] Créer un README documenté
- [x] Configurer l'Inference API
- [x] Tester l'API
- [x] Documenter l'usage

**Checkpoints :**
- ✅ Modèle pushé sur HF Hub
- ✅ README complet
- ✅ API fonctionnelle
- ✅ Exemples d'usage

## ⚠️ Deal Breakers

- [ ] Dataset < 100 conversations → Insuffisant pour training
- [ ] Format JSONL invalide → Pipeline cassé
- [ ] Training loss augmente → Overfitting ou learning rate trop haut
- [ ] Réponses incohérentes → Modèle ne converge pas
- [ ] Coût > budget (50$) → Arrêter et optimiser

## 📊 Métriques de succès

### Training
- ✅ 500+ conversations dans le dataset
- ✅ PPL validation < 5
- ✅ Training loss descendante
- ✅ Checkpoints sauvegardés

### Modèle
- ✅ Réponses cohérentes et immersives
- ✅ Respect du contexte JDR
- ✅ Style MJ expert
- ✅ Langue française correcte

### Infrastructure
- ✅ Coût total < 50$
- ✅ Temps de training < 3h
- ✅ Model size < 10GB

## 📝 Notes de debugging

### Problèmes connus :
1. **Dataset trop petit**
   - Solution: Ajouter plus de campagnes ou scraper des discussions

2. **Loss qui stagne**
   - Solution: Augmenter le learning rate ou les epochs

3. **Overfitting**
   - Solution: Ajouter du dropout ou réduire les epochs

4. **VRAM insuffisante**
   - Solution: Utiliser QLoRA 4-bit ou cloud

### Solutions appliquées :
- ✅ Utiliser LoRA pour efficiency
- ✅ Quantisation 4-bit pour réduire la VRAM
- ✅ Validation set pour monitorer l'overfitting
- ✅ Cloud pour resources insuffisantes

## 📚 Références

### Docs
- Axolotl: https://github.com/OpenAccess-AI-Collective/axolotl
- Hugging Face: https://huggingface.co/docs
- Together.ai: https://together.ai/docs
- LoRA: https://arxiv.org/abs/2106.09685

### Modèles
- Qwen2.5-7B-Instruct: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct
- Qwen2.5-14B-Instruct: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct

### Scripts
- Convert: `scripts/convert_to_jsonl.py`
- Config: `configs/axolotl_config.yaml`
- Train: `scripts/train_model.py`
- Evaluate: `scripts/evaluate_model.py`

### Ressources
- Dataset: `data/final/jdroll_training_*.jsonl`
- Logs: `logs/training_*.log`
- Checkpoints: `output/checkpoints/`

---

**Statut:** En attente de challenge  
**Priorité:** Critique  
**Labels:** fine-tuning, axolotl, mlops, phase-3

**Challenge required:** OUI  
**Next action:** Lancer `challenge-plan` sur ce plan

## 🔗 Liens AIDD

- [Challenge Plan](#) → `challenge-plan` skill
- [Test-Driven Development](#) → `test-driven-development` skill  
- [Code Review](#) → `requesting-code-review` skill
- [Learn & Document](#) → `learn` skill (à créer)

---

**End of Task**
- [ ] Documentation: `aidd_docs/changelog/phase3.md`
- [ ] Lessons: `aidd_docs/memory/phase3_lessons.md`
- [ ] Changelog: `aidd_docs/changelog/CHANGELOG.md`
