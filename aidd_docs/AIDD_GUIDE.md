# 🚀 Guide Complet AIDD - AI-Driven Development

## 📋 Introduction

Le framework **AIDD** (AI-Driven Development) est un système complet pour gérer les configurations, commandes, règles et templates pour les outils de développement IA.

### 🔗 Dépôts Officiels

| Dépôt | URL | Description |
|-------|-----|-------------|
| **CLI** | https://github.com/RebelliousSmile/aidd-custom | CLI pour installer les configurations AIDD |
| **Overlay** | https://github.com/RebelliousSmile/aidd-overlay | Contenu personnalisé (commandes, règles, templates) |

---

## 📦 Installation

### 1️⃣ Installer le CLI

```bash
# Via npm
npm install -g aidd-custom

# Vérifier l'installation
aidd-custom --version
```

### 2️⃣ Configuration initiale

```bash
# Aller dans le projet
cd /home/user/suddenly-muses

# Configurer le framework (pointe vers l'overlay)
aidd-custom setup --repo RebelliousSmile/aidd-overlay
```

---

## 🛠 Installation des Outils

### ⚠️ IMPORTANT : Choix de l'outil

Le CLI supporte plusieurs outils, mais **leurs fonctionnements diffèrent** :

#### **🎯 RECOMMANDÉ : Copilot**
```bash
aidd-custom install --ai copilot
```
- ✅ Fonctionnement le plus proche du framework AIDD
- ✅ Fichiers optimisés
- ✅ Meilleure intégration

#### **⚙️ Autres outils (fonctionnement différent)**

```bash
# Claude Code - Configuration spécifique
aidd-custom install --ai claude

# Cursor - Configuration spécifique  
aidd-custom install --ai cursor

# Opencode - Configuration spécifique
aidd-custom install --ai opencode
```

**Note :** Claude, Cursor et Opencode ont des fonctionnements différents et peuvent nécessiter des ajustements.

### 🔧 Installation multiple

```bash
# Installer plusieurs outils
aidd-custom install --ai copilot claude cursor opencode

# Ou un par un (recommandé pour copilot)
aidd-custom install --ai copilot
aidd-custom install --ai claude
aidd-custom install --ai cursor
aidd-custom install --ai opencode
```

---

## 📚 Structure des Fichiers

### 📁 aidd-overlay (Contenu)

```
aidd-overlay/
├── agents/           # Agents Claude Code
├── commands/         # Slash commands (/...)
├── memory/           # Mémoire externe
├── misc/             # Ressources optionnelles
├── rules/            # Règles Claude Code
├── skills/           # Skills Claude Code
├── templates/        # Templates
│   └── dev/          # Templates techniques
└── .claude-plugin/   # Métadonnées plugin
```

### 📁 Installation finale

```
.
├── .claude/
│   ├── commands/     # Commands installés
│   ├── rules/        # Rules installées
│   ├── agents/       # Agents installés
│   └── templates/    # Templates installés
├── .opencode/
│   ├── commands/
│   └── rules/
├── .cursor/
│   └── rules/
└── .github/
    └── copilot/      # Copilot configurations
```

---

## 🔄 Commandes du CLI

### Configuration

```bash
# Voir la configuration actuelle
aidd-custom setup

# Configurer un nouvel overlay
aidd-custom setup --repo RebelliousSmile/aidd-overlay

# Changer de branche
aidd-custom setup --repo RebelliousSmile/aidd-overlay --branch main
```

### Installation

```bash
# Installer pour un outil spécifique
aidd-custom install --ai copilot

# Installer pour plusieurs outils
aidd-custom install --ai copilot claude

# Installer tous les outils AI
aidd-custom install --ai --all

# Installer les outils IDE
aidd-custom install --ide vscode
```

### Maintenance

```bash
# Vérifier l'installation
aidd-custom doctor

# Mettre à jour
aidd-custom update

# Nettoyer
aidd-custom clean

# Voir le statut
aidd-custom status

# Synchroniser les modifications locales
aidd-custom sync
```

---

## 📋 Workflow AIDD

### 1️⃣ Brainstorming
- Analyser le besoin
- Identifier les fichiers nécessaires

### 2️⃣ Création d'issue
- Créer une issue GitHub avec `aidd-custom`
- Lister les tâches et critères d'acceptation

### 3️⃣ Planification
- Créer un plan AIDD (`aidd_docs/tasks/phaseX_*.md`)
- Inclure les critères d'acceptation

### 4️⃣ Challenge
- Valider le plan avec `aidd-custom`
- Score attendu: 9.0/10 minimum

### 5️⃣ Implémentation
- Développer selon le plan
- Tests obligatoires avant commit

### 6️⃣ Code Review
- Vérifier la conformité AIDD
- S'assurer que les templates sont respectés

### 7️⃣ Functional Review
- Tester les fonctionnalités
- Vérifier les critères d'acceptation

### 8️⃣ Commit & Finalisation
- Commit propre avec message conventionnel
- Mettre à jour le changelog

### 9️⃣ Lessons Learned
- Documenter les apprentissages
- `aidd_docs/memory/lessons_phaseX.md`

### 🔟 Changelog
- Mettre à jour `CHANGELOG.md`
- Versionner selon [Keep a Changelog](https://keepachangelog.com/)

---

## 🎯 Exemples d'Utilisation

### Exemple 1 : Configuration complète

```bash
# Aller dans le projet
cd /home/user/suddenly-muses

# Installer le CLI
npm install -g aidd-custom

# Configurer l'overlay
aidd-custom setup --repo RebelliousSmile/aidd-overlay

# Installer pour Copilot (recommandé)
aidd-custom install --ai copilot

# Vérifier l'installation
aidd-custom doctor
```

### Exemple 2 : Workflow AIDD complet

```bash
# 1. Créer un plan pour une nouvelle feature
aidd_custom_plan create --name "feature-x" --issue 50

# 2. Challenger le plan
aidd_custom_plan challenge feature-x.md

# 3. Implémenter (manuellement ou avec assistance AI)

# 4. Documenter les lessons
aidd_custom_learn --phase 3

# 5. Mettre à jour le changelog
vim aidd_docs/changelog/CHANGELOG.md
```

### Exemple 3 : Maintenance

```bash
# Mettre à jour vers la dernière version
aidd-custom update

# Nettoyer les anciennes installations
aidd-custom clean

# Voir ce qui est installé
aidd-custom status
```

---

## 📊 Comparaison des Outils

| Outil | Fonctionnement | Performance | Recommandation |
|-------|---------------|-------------|----------------|
| **Copilot** | Standard AIDD | ⭐⭐⭐⭐⭐ | ✅ **Recommandé** |
| **Claude Code** | Personnalisé | ⭐⭐⭐⭐ | ⚠️ Bon mais différent |
| **Cursor** | Personnalisé | ⭐⭐⭐⭐ | ⚠️ Bon mais différent |
| **Opencode** | Personnalisé | ⭐⭐⭐ | ⚠️ Bon mais différent |

**Pourquoi Copilot est recommandé :**
- ✅ Fonctionnement le plus proche du framework
- ✅ Meilleure compatibilité avec les templates
- ✅ Fichiers générés optimisés
- ✅ Support officiel du framework

---

## 🔐 Sécurité & Bonnes Pratiques

### 1. Never Commit Secrets
```bash
# .env NE JAMAIS commité
*.env
*.env.local
*.env.*.local
```

### 2. Token GitHub
```bash
# Utiliser un token GitHub PAT (Fine-grained)
# Permissions: repo + pull_requests
# Stocker dans .env
```

### 3. Version Control
```bash
# Toujours versionner les modifications AIDD
git add aidd_docs/
git commit -m "docs: update AIDD framework"
```

---

## 📝 FAQ

### Q: Pourquoi utiliser `--ai copilot` ?
**R:** Copilot a un fonctionnement plus proche du framework AIDD que Claude, Cursor ou Opencode qui ont des configurations spécifiques.

### Q: Comment mettre à jour l'overlay ?
**R:** `aidd-custom update` ou `aidd-custom setup --repo RebelliousSmile/aidd-overlay`

### Q: Les fichiers AIDD peuvent-ils être supprimés manuellement ?
**R:** Non, utilisez toujours `aidd-custom clean` ou `aidd-custom uninstall`.

### Q: Comment ajouter une nouvelle command ?
**R:** Ajouter dans `aidd-overlay/commands/` puis `aidd-custom install --ai copilot`.

### Q: Les sous-modules aidd-custom/aidd-overlay sont-ils nécessaires ?
**R:** Oui, ils contiennent le code source. Ils peuvent être retirés une fois le CLI installé globalement.

---

## 🔄 Migration depuis l'ancienne méthode

### Ancienne méthode (obsolète)
```bash
# ❌ À NE PLUS UTILISER
npm install -g @ai-driven-dev/cli
aidd setup --repo ai-driven-dev/aidd-framework
aidd install ai --all
```

### Nouvelle méthode (recommandée)
```bash
# ✅ METHODE CORRECTE
npm install -g aidd-custom
aidd-custom setup --repo RebelliousSmile/aidd-overlay
aidd-custom install --ai copilot
```

---

## 📚 Ressources

- **GitHub CLI:** https://github.com/RebelliousSmile/aidd-custom
- **GitHub Overlay:** https://github.com/RebelliousSmile/aidd-overlay
- **Documentation AIDD:** https://hermes-agent.nousresearch.com/docs/user-guide/features/memory
- **Workflow AIDD:** [`aidd_docs/WORKFLOW.md`](./aidd_docs/WORKFLOW.md)

---

## 🎯 Conclusion

Le framework AIDD avec **aidd-custom** et **aidd-overlay** offre :
- ✅ Gestion centralisée des configurations
- ✅ Templates et règles réutilisables
- ✅ Automatisation des installations
- ✅ Support multi-outils (mais Copilot recommandé)
- ✅ Workflow structuré (Brainstorm → Plan → Implémenter → Review → Lessons)

**Pour un résultat optimal :**
1. Utiliser `--ai copilot` pour l'installation
2. Suivre le workflow AIDD
3. Documenter systématiquement
4. Maintenir le framework à jour

---

*Dernière mise à jour: 2026-05-13*
*Version: 1.0.0*
