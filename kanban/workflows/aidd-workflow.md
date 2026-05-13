# 🔄 Workflow AIDD

> Guide complet du framework AI-Driven Development

## 📋 Overview

Le workflow AIDD structure le développement en 5 étapes principales :

1. **Task** : Création de la tâche
2. **Plan** : Écriture du plan détaillé
3. **Challenge** : Revue et validation du plan
4. **Execute** : Exécution des tâches
5. **Review** : Validation et documentation

## 🚀 Démarrage

### **1. Créer une tâche**

```bash
# Via CLI AIDD
aidd task create --name "feature-x" --phase 3 --priority high

# Ou manuellement
# Créer kanban/cards/phase-3/task-feature-x.md
```

### **2. Créer un plan**

```bash
# Via compétence
hermes -s writing-plans "Créer un plan pour feature-x"

# Ou via CLI
aidd plan create --task task-feature-x.md
```

### **3. Décomposer le plan**

```bash
# Décomposition automatique
aidd decompose --plan plan-feature-x.md

# Ou manuellement
# Utiliser la compétence 'plan' dans Hermes
```

### **4. Challenge et validation**

```bash
# Validation par challenge
aidd challenge --plan plan-feature-x.md --review reviews/challenge.md

# Ou via compétences
hermes -s challenge-plan "Valider le plan feature-x"
```

### **5. Exécution**

```bash
# Exécuter les tâches une par une
# Documenter les progrès dans le plan
# Créer des notes dans la section "execution-notes"
```

### **6. Review et documentation**

```bash
# Validation finale
aidd review --plan plan-feature-x.md

# Documentation
hermes -s aidd-workflow "Créer un review pour feature-x"
```

## 📝 Templates AIDD

### **Plan AIDD**

```markdown
# Plan: <NOM_DU_PLAN>

## 📋 Contexte
<DESCRIPTION_DU_CONTEXTE>

## 🎯 Objectifs
- <OBJECTIF_1>
- <OBJECTIF_2>

## 📝 Tâches

### 1. <NOM_DE_LA_TACHE_1>
**Statut** : pending | in_progress | completed
**Temps estimé** : <X>h
**Description** : <DESCRIPTIF>
**Acceptance criteria** :
- [ ] <CRITERE_1>
- [ ] <CRITERE_2>

### 2. <NOM_DE_LA_TACHE_2>
...

## 🧩 Dépendances
- <TACHE_DEP_1>
- <TACHE_DEP_2>

## ⚠️ Risques
- <RISQUE_1>
- <RISQUE_2>

## 📚 Références
- [Lien 1](<URL_1>)

## 📊 Progression
- [ ] <TACHE_1>
- [ ] <TACHE_2>
- [ ] <TACHE_3>

## 📝 Notes d'exécution
<NOTE_1>
<NOTE_2>
```

### **Challenge AIDD**

```markdown
# Challenge: <NOM_DU_PLAN>

## 🎯 Objectif du challenge
<OBJECTIF>

## 📋 Critères de validation
- [ ] <CRITERE_1>
- [ ] <CRITERE_2>
- [ ] <CRITERE_3>

## 🔍 Points d'attention
- <POINT_1>
- <POINT_2>

## ✅ Résultat
- **Validé** : OUI/NON
- **Commentaires** : <COMMENTAIRES>
- **Recommandations** : <RECOMMANDATIONS>

## 📊 Métriques
- **Qualité du plan** : <1-5>
- **Clarté** : <1-5>
- **Complétude** : <1-5>
- **Faisabilité** : <1-5>
```

## 🔄 Intégration Hermes Agent

### **Compétences AIDD**

```bash
# Workflow complet
hermes -s aidd-workflow "Workflow AIDD"

# Planification
hermes -s writing-plans "Créer un plan structuré"
hermes -s plan "Planification de tâche"

# Validation
hermes -s challenge-plan "Challenge et validation"

# Documentation
hermes -s learn "Documentation et lessons learned"
```

### **Exemple d'utilisation**

```bash
# 1. Créer un plan
hermes -s writing-plans "Créer un plan pour phase 3 - Fine-tuning"

# 2. Décomposer
hermes -s plan "Décomposer le plan en tâches"

# 3. Challenge
hermes -s challenge-plan "Valider le plan de phase 3"

# 4. Exécuter
# (exécuter manuellement ou via scripts)

# 5. Review
hermes -s aidd-workflow "Créer un review pour phase 3"
```

## 📚 Références

- [Documentation AIDD](./aidd_docs/)
- [Workflow AIDD](./aidd_docs/WORKFLOW.md)
- [Guide AIDD](./aidd_docs/AIDD_GUIDE.md)
