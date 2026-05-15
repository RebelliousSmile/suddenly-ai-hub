# 🚀 Installation Rapide - Suddenly AI Hub

> Guide d'installation en 5 minutes

## ⏱️ Time to Hello World

**~5 minutes** pour avoir tout configuré et prêt à l'emploi !

## 📋 Prérequis

- ✅ Python 3.11+ installé
- ✅ Node.js 18+ installé
- ✅ npm disponible
- ✅ Accès GitHub (pour cloner les dépôts)

## 🚀 Installation en 3 étapes

### **Étape 1 : Installer le CLI AIDD**

```bash
# Aller dans suddenly-ai-hub
cd /home/user/suddenly-ai-hub

# Installer aidd-custom
cd aidd-custom
npm install
npm link

# Vérifier
aidd-custom --version
```

### **Étape 2 : Configurer le framework**

```bash
# Retourner au projet
cd /home/user/suddenly-ai-hub

# Configurer l'overlay
aidd-custom setup --repo RebelliousSmile/aidd-overlay

# Installer les configurations
aidd-custom install

# Vérifier
aidd-custom doctor
```

### **Étape 3 : Installer les compétences Hermes**

```bash
# Les compétences sont dans ~/.hermes/plugins/hermes-vault/
# Elles sont déjà liées vers ~/.hermes/skills/

# Redémarrer Hermes Agent pour les scanner
# (Fermer et rouvrir l'interface)

# Vérifier les compétences
hermes skills list
```

## ✅ Vérification

```bash
# 1. CLI AIDD
aidd-custom --version

# 2. Framework
aidd-custom doctor

# 3. Compétences
hermes skills list | grep aidd

# 4. Projet
ls -la suddenly-ai-hub/
```

## 🔧 Dépannage

### **Problème : aidd-custom n'est pas trouvé**

```bash
# Ajouter au PATH
export PATH="$HOME/.npm-global/bin:$PATH"

# Ou utiliser le chemin complet
/home/user/suddenly-ai-hub/aidd-custom/dist/cli.js --help
```

### **Problème : Compétences non détectées**

```bash
# Redémarrer Hermes Agent
# Fermer et rouvrir l'interface

# Forcer le scan
hermes skills reload  # Si disponible
```

### **Problème : Permissions npm**

```bash
# Si erreur d'installation npm
sudo chown -R $(whoami) ~/.npm
```

## 📚 Étapes suivantes

1. ✅ Installer les outils (cette page)
2. 📖 Lire [README.md](./README.md)
3. 🎯 Lire [aidd_docs/](./aidd_docs/)
4. 🚀 Commencer Phase 3 (fine-tuning)

## 🎯 Commandes utiles

```bash
# AIDD CLI
aidd-custom setup --repo RebelliousSmile/aidd-overlay
aidd-custom install
aidd-custom doctor
aidd-custom clean

# Compétences Hermes
hermes skills list
hermes skills inspect <skill-name>
hermes -s <skill-name> "commande"

# Projet
git status
git pull origin main
```

## 📊 Statistiques d'installation

| Élément | Statut | Détails |
|---------|--------|---------|
| **CLI AIDD** | ✅ | `aidd-custom` v1.0.0 |
| **Framework** | ✅ | 112 fichiers installés |
| **Compétences** | ✅ | 11 compétences custom |
| **Documentation** | ✅ | 44 fichiers AIDD |

## 🔐 Variables d'environnement

Créer `.env` dans le projet :

```bash
# GitHub
GITHUB_TOKEN=votre_token

# Together.ai
TOGETHER_API_KEY=votre_key

# Fireworks.ai
FIREWORKS_API_KEY=votre_key

# Weights & Biases
WANDB_API_KEY=votre_key
```

## 📞 Besoin d'aide ?

- **Documentation complète** : [README.md](./README.md)
- **Workflow AIDD** : [aidd_docs/WORKFLOW.md](./aidd_docs/WORKFLOW.md)
- **Compétences** : [hermes-vault](https://github.com/RebelliousSmile/hermes-vault)
- **GitHub** : [suddenly-ai-hub](https://github.com/RebelliousSmile/suddenly-ai-hub)

---

**Installation terminée en ~5 minutes ! 🎉**
