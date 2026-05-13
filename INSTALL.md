# 🚀 Installation Rapide - AIDD Suddenly AI Hub

> Guide d'installation en 5 minutes pour commencer à utiliser le framework AIDD

## 📋 Prérequis

- ✅ Node.js 18+ installé
- ✅ npm disponible
- ✅ Accès internet

## 📦 Installation (2 minutes)

### 1️⃣ Installer le CLI

```bash
# Installer aidd-custom globalement
npm install -g aidd-custom

# Vérifier l'installation
aidd-custom --version
# → Doit afficher: aidd-custom/1.0.0+ (ou similaire)
```

### 2️⃣ Configurer le Projet

```bash
# Aller dans suddenly-ai-hub
cd /home/user/suddenly-ai-hub

# Configurer l'overlay
aidd-custom setup --repo RebelliousSmile/aidd-overlay

# Installer pour Copilot (recommandé)
aidd-custom install --ai copilot
```

### 3️⃣ Vérifier l'Installation

```bash
# Voir le statut
aidd-custom status

# Vérifier la santé
aidd-custom doctor
```

## 🎯 Utilisation Immédiate

### Workflow de Base

```bash
# 1. Créer un nouveau plan pour une feature
aidd-custom plan create --name "feature-x" --issue 50

# 2. Challenger le plan
aidd-custom plan challenge phase3_finetuning.md

# 3. Implémenter (avec assistance AI)

# 4. Documenter les lessons
aidd-custom learn --phase 3

# 5. Mettre à jour le changelog
vim aidd_docs/changelog/CHANGELOG.md
```

## 📁 Structure du Projet

```
suddenly-ai-hub/
├── .claude/              # Configurations Claude
├── .opencode/            # Configurations Opencode
├── aidd_docs/            # Documentation AIDD
│   ├── tasks/            # Plans de phases
│   ├── reviews/          # Challenges
│   ├── memory/           # Mémoire
│   └── changelog/        # CHANGELOG
└── scripts/              # Scripts Python
```

## 📚 Commandes Essentielles

| Commande | Description |
|----------|-------------|
| `aidd-custom setup` | Configurer le framework |
| `aidd-custom install --ai copilot` | Installer pour Copilot |
| `aidd-custom doctor` | Vérifier l'installation |
| `aidd-custom update` | Mettre à jour |
| `aidd-custom clean` | Nettoyer |
| `aidd-custom status` | Voir le statut |

## 🌐 Liens Rapides

- **CLI:** https://github.com/RebelliousSmile/aidd-custom
- **Overlay:** https://github.com/RebelliousSmile/aidd-overlay
- **Guide Complet:** `aidd_docs/AIDD_GUIDE.md`
- **Workflow:** `aidd_docs/WORKFLOW.md`

## ⚠️ Points Importants

### 1️⃣ Pour Copilot vs Autres Outils

```bash
# ✅ RECOMMANDÉ (fonctionnement le plus proche)
aidd-custom install --ai copilot

# ⚠️ FONCTIONNEMENT DIFFÉRENT (nécessite ajustements)
aidd-custom install --ai claude
aidd-custom install --ai cursor
aidd-custom install --ai opencode
```

### 2️⃣ Sécurité

- ❌ NE JAMAIS commit de `.env`
- ✅ Stocker les tokens dans `~/.hermes/.env`
- ✅ Utiliser des tokens GitHub avec permissions minimales

### 3️⃣ Maintenance

```bash
# Mettre à jour le framework
aidd-custom update

# Nettoyer les anciennes installations
aidd-custom clean

# Synchroniser les modifications locales
aidd-custom sync
```

## 🎉 Premières Étapes

Une fois installé, tu peux :

1. ✅ **Lire la documentation:** `aidd_docs/AIDD_GUIDE.md`
2. ✅ **Voir le workflow:** `aidd_docs/WORKFLOW.md`
3. ✅ **Implémenter Phase 3:** `aidd_docs/tasks/phase3_finetuning.md`
4. ✅ **Utiliser le framework** avec tes outils IA préférés

## 🐛 Dépannage

### Problème: "command not found: aidd-custom"

```bash
# Solution 1: Réinstaller
npm install -g aidd-custom

# Solution 2: Vérifier le PATH
echo $PATH | grep node_modules

# Solution 3: Redémarrer le terminal
```

### Problème: "Failed to connect to GitHub"

```bash
# Vérifier la connexion
curl -I https://github.com

# Vérifier les permissions du token
gh auth status
```

### Problème: "Overlay not configured"

```bash
# Réconfigurer
aidd-custom setup --repo RebelliousSmile/aidd-overlay
```

## 📞 Support

- **GitHub Issues:** https://github.com/RebelliousSmile/aidd-custom/issues
- **Documentation:** `aidd_docs/`

---

**Installation complète en < 5 minutes !** ⚡

*Dernière mise à jour: 2026-05-13*
