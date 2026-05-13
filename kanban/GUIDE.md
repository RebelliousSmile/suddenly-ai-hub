# 📚 Guide du Kanban

> Comment utiliser le kanban Suddenly AI Hub

## 🎯 Objectif

Ce kanban structure le développement du projet en suivant le framework AIDD (AI-Driven Development).

## 📋 Structure

```
kanban/
├── README.md          ← Ce guide
├── DASHBOARD.md       ← Tableau de bord
├── cards/             ← Cartes de tâches
├── workflows/         ← Workflows de processus
└── templates/         ← Templates
```

## 🔄 Processus standard

### **1. Création d'une nouvelle tâche**

1. Créer une carte dans `kanban/cards/phase-X/`
2. Utiliser le template `kanban/templates/task-template.md`
3. Remplir tous les champs obligatoires
4. Commiter et pusher

### **2. Développement d'une tâche**

1. Lier la tâche à un issue GitHub (si applicable)
2. Documenter les progrès dans la carte
3. Mettre à jour le statut et les sous-tâches
4. Utiliser `aidd-workflow` pour structurer

### **3. Review et validation**

1. Suivre `kanban/workflows/review-workflow.md`
2. Utiliser `challenge-plan` compétence Hermes
3. Valider les critères d'acceptation
4. Archiver la tâche si terminée

### **4. Documentation**

1. Ajouter les lessons learned dans `aidd_docs/memory/`
2. Mettre à jour `DASHBOARD.md`
3. Créer un review dans `aidd_docs/reviews/`

## 📝 Bonnes pratiques

### **Règles de naming**

- Tâches : `TASK-X.Y: Description` (ex: `TASK-3.1: Analyse des données`)
- Épics : `EPIC-X: Nom` (ex: `EPIC-3: Fine-tuning`)
- Phases : `PHASE-X: Nom` (ex: `PHASE-3: Fine-tuning`)

### **Règles de statut**

- `pending` : Pas encore commencé
- `in_progress` : En cours
- `completed` : Terminé
- `blocked` : Bloqué (avec commentaire)

### **Règles de priorité**

- `high` : Critique, doit être fait maintenant
- `medium` : Important, peut attendre
- `low` : Nice to have

### **Règles de métadonnées**

- Toujours mettre à jour `modified` à chaque changement
- Documenter les blockers dans les commentaires
- Lier aux issues GitHub quand applicable

## 🚀 Démarrage rapide

```bash
# Créer une nouvelle tâche
hermes -s writing-plans "Créer une tâche pour feature-x"

# Utiliser le template
cp kanban/templates/task-template.md kanban/cards/task-3.6-feature-x.md

# Mettre à jour le dashboard
hermes -s aidd-workflow "Mettre à jour DASHBOARD.md"
```

## 📚 Compétences associées

- `aidd-workflow` : Workflow AIDD
- `writing-plans` : Création de plans
- `plan` : Planification
- `challenge-plan` : Validation
- `learn` : Documentation

## 🔗 Liens

- [Workflow AIDD](./workflows/aidd-workflow.md)
- [Workflow Review](./workflows/review-workflow.md)
- [Documentation AIDD](../aidd_docs/)
- [README](../README.md)
