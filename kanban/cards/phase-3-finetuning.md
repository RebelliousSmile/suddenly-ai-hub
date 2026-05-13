---
id: PHASE-3
name: Phase 3 - Fine-tuning Modèle JDR
phase: 3
status: in_progress
priority: high
owner: RebelliousSmile
created: 2026-05-13
modified: 2026-05-13
description: Fine-tuning d'un modèle de langage pour la génération de dialogues JDR en français
objectives:
  - Créer un dataset de qualité pour le fine-tuning
  - Configurer et lancer le fine-tuning
  - Évaluer et itérer sur les résultats
success_metrics:
  - Modèle capable de générer des dialogues JDR cohérents
  - Qualité supérieure à celle du modèle de base
  - Latence acceptable pour l'interaction
tasks:
  - TASK-3.1
  - TASK-3.2
  - TASK-3.3
  - TASK-3.4
  - TASK-3.5
dependencies:
  - PHASE-2: Scraping des données (Session 2)
risks:
  - Données insuffisantes ou de mauvaise qualité
  - Ressources GPU limitées
  - Modèle ne converge pas correctement
---

# 🎯 Phase 3 - Fine-tuning Modèle JDR

## 📝 Description

Cette phase vise à fine-tuner un modèle de langage (ex: Mistral 7B) avec des données JDR en français pour améliorer sa capacité à générer des dialogues et narrations JDR.

## 🎯 Objectifs

- [x] Analyser les données disponibles (Session 2)
- [ ] Préparer et nettoyer le dataset
- [ ] Configurer le pipeline de fine-tuning (Axolotl/Unsloth)
- [ ] Lancer le fine-tuning
- [ ] Évaluer les résultats et itérer

## 📊 Métriques de succès

- [x] Dataset JDR prêt (>10k exemples)
- [ ] Modèle fine-tuné avec BLEU/ROUGE améliorés
- [ ] Latence <2s pour la génération
- [ ] Cohérence des dialogues JDR

## 🧩 Tâches principales

### **TASK-3.1: Analyse des données**

**Statut** : ✅ Terminé  
**Assigné** : RebelliousSmile  
**Temps estimé** : 2h

**Description** :
Analyser les données disponibles dans Session 2 pour identifier la qualité, la quantité et le format.

**Acceptance criteria** :
- [x] Dataset analysé
- [x] Rapport de qualité généré
- [x] Données nettoyées et formatées

---

### **TASK-3.2: Préparation du dataset**

**Statut** : ⏸️ En attente  
**Assigné** : RebelliousSmile  
**Temps estimé** : 4h

**Description** :
Préparer et nettoyer le dataset pour le fine-tuning.

**Acceptance criteria** :
- [ ] Données converties au format JSONL
- [ ] Dataset équilibré (dialogues/narration)
- [ ] Données annotées avec tags
- [ ] Split train/val/test (80/10/10)

---

### **TASK-3.3: Configuration du pipeline**

**Statut** : ⏸️ En attente  
**Assigné** : RebelliousSmile  
**Temps estimé** : 3h

**Description** :
Configurer le pipeline de fine-tuning avec Axolotl ou Unsloth.

**Acceptance criteria** :
- [ ] Configuration YAML complète
- [ ] Parameters optimisés pour JDR
- [ ] Tracking avec WandB configuré
- [ ] Scripts de training prêts

---

### **TASK-3.4: Exécution du fine-tuning**

**Statut** : ⏸️ En attente  
**Assigné** : RebelliousSmile  
**Temps estimé** : 24h+

**Description** :
Lancer le fine-tuning et suivre les métriques.

**Acceptance criteria** :
- [ ] Fine-tuning lancé avec succès
- [ ] Métriques trackingées (WandB)
- [ ] Checkpoints sauvegardés
- [ ] Modèle final exporté

---

### **TASK-3.5: Évaluation et itération**

**Statut** : ⏸️ En attente  
**Assigné** : RebelliousSmile  
**Temps estimé** : 8h

**Description** :
Évaluer le modèle fine-tuné et itérer si nécessaire.

**Acceptance criteria** :
- [ ] Évaluation quantitative (BLEU, ROUGE)
- [ ] Évaluation qualitative (dialogues tests)
- [ ] Rapports de performance
- [ ] Itérations si nécessaire

## 🔗 Dépendances

- **PHASE-2** : Scraping des données (Session 2)

## ⚠️ Risques identifiés

1. **Données insuffisantes** : Risque moyen
   - Mitigation : Scraping supplémentaire, augmentation de données

2. **Ressources GPU limitées** : Risque faible
   - Mitigation : Utilisation d'Unsloth, quantisation

3. **Non-convergence** : Risque moyen
   - Mitigation : Ajustement des hyperparamètres, augmentation du dataset

## 📚 Références

- [Guide Axolotl](https://github.com/OpenAccess-AI-Collective/axolotl)
- [Guide Unsloth](https://github.com/unslothai/unsloth)
- [Documentation AIDD](./aidd_docs/)

## 📊 Progression

| Tâche | Statut | Progression |
|-------|--------|-------------|
| TASK-3.1: Analyse des données | ✅ Terminé | 100% |
| TASK-3.2: Préparation du dataset | ⏸️ En attente | 0% |
| TASK-3.3: Configuration du pipeline | ⏸️ En attente | 0% |
| TASK-3.4: Exécution du fine-tuning | ⏸️ En attente | 0% |
| TASK-3.5: Évaluation et itération | ⏸️ En attente | 0% |

---

**Propriétaire** : RebelliousSmile  
**Phase** : 3 - Fine-tuning  
**Statut** : En préparation  
**Priorité** : Haute
