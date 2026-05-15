# 🚀 Suddenly AI Hub

> **Fine-tuning d'un modèle de langage JDR (Jeux de Rôle) en français**

## 📋 Résumé

Ce projet vise à fine-tuner un modèle de langage pour générer des dialogues et narrations JDR en français, spécialisé dans l'univers [UNIVERS JDR] (ex: Cyberpunk, Fantasy, Horreur, etc.).

## 🎯 Objectifs

1. **Phase 1** : Configuration et setup
2. **Phase 2** : Collecte et préparation des données (scraping)
3. **Phase 3** : Fine-tuning du modèle JDR
4. **Phase 4** : Amélioration continue et évaluation
5. **Phase 5** : Scaling vers 100+ campagnes
6. **Phase 6** : Production et déploiement
7. **Phase 7** : Communauté et sharing

## 📦 Installation

### **Prérequis**

- Python 3.11+
- Node.js 18+
- npm
- GPU (recommandé pour le fine-tuning)

### **Installation de aidd-custom**

```bash
# Installer le CLI AIDD
cd /home/user/suddenly-ai-hub/aidd-custom
npm install
npm link

# Configurer le framework
aidd-custom setup --repo RebelliousSmile/aidd-overlay

# Installer les configurations
aidd-custom install
```

### **Installation des compétences Hermes**

```bash
# Installer les compétences custom
cd /home/user/hermes-skills
npm install
npm link

# Redémarrer Hermes Agent
# Les compétences seront scannées automatiquement
```

## 🏗️ Architecture

### **Structure du projet**

```
suddenly-ai-hub/
├── .claude/                # Configurations Claude
├── .copilot/              # Configurations Copilot
├── .cursor/               # Configurations Cursor
├── .opencode/             # Configurations Opencode
├── aidd_docs/             # Documentation AIDD
│   ├── tasks/             # Plans de tâches
│   ├── reviews/           # Challenges validés
│   ├── changelog/         # CHANGELOG.md + VERSIONS.md
│   └── memory/            # Templates et mémoire
├── scripts/               # Scripts de fine-tuning
│   ├── convert_to_jsonl.py
│   └── train_model.py
├── data/                  # Données (à créer)
├── models/                # Modèles (à créer)
├── results/               # Résultats (à créer)
├── INSTALL.md             # Guide d'installation
├── README.md              # Ce fichier
└── LICENSE                # Licence
```

## 🛠️ Outils utilisés

- **Framework AIDD** : Gestion des configurations et workflows
- **Hermes Agent** : Compétences custom (11 compétences)
- **Axolotl** : Fine-tuning
- **Unsloth** : Fine-tuning optimisé
- **Axolotl** : Configuration de fine-tuning
- **WandB** : Tracking des expériences

## 📋 Compétences AIDD

### **Développement**
- `aidd-workflow` : Workflow AIDD complet
- `challenge-plan` : Challenge et validation des plans
- `writing-plans` : Écriture de plans structurés
- `plan` : Planification de tâches

### **DevOps**
- `aidd-workflow` : Workflow AIDD
- `challenge-plan` : Validation des plans
- `jdroll-data-pipeline` : Pipeline de données JDR
- `learn` : Documentation des lessons learned

### **Mlops**
- `fine-tuning-roleplay-rp` : Fine-tuning pour rôleplay
- `fine-tuning-domain-dialogue` : Fine-tuning domain-specific

### **GitHub**
- `github-auth` : Authentification GitHub
- `github-pr-workflow` : Workflow PR

## 📊 Progression

| Phase | Statut | Description |
|-------|--------|-------------|
| **Phase 1** | ✅ Terminé | Setup et configuration |
| **Phase 2** | ⏸️ En attente | Scraping des données (Session 2) |
| **Phase 3** | 🔧 En préparation | Fine-tuning modèle JDR |
| **Phase 4** | 📋 Planifié | Amélioration continue |
| **Phase 5** | 📋 Planifié | Scaling vers 100+ campagnes |
| **Phase 6** | 📋 Planifié | Production |
| **Phase 7** | 📋 Planifié | Communauté |

## 🚀 Démarrage rapide

### **1. Installer les outils**

```bash
# CLI AIDD
npm install -g aidd-custom
aidd-custom setup --repo RebelliousSmile/aidd-overlay
aidd-custom install

# Vérifier
aidd-custom doctor
```

### **2. Utiliser le framework**

```bash
# Créer un plan
aidd plan create --name "feature-x" --issue 50

# Examiner un plan
aidd plan examine --plan plan.md

# Décomposer un plan
aidd decompose --plan plan.md

# Valider un plan
aidd validate --plan plan.md --challenge challenges.md
```

### **3. Commencer le fine-tuning**

```bash
# Convertir les données
python scripts/convert_to_jsonl.py

# Lancer le fine-tuning
python scripts/train_model.py --model qwen2.5_7b

# Track avec WandB
wandb login
```

## 📚 Documentation

- **[INSTALL.md](./INSTALL.md)** : Guide d'installation rapide
- **[aidd_docs/](./aidd_docs/)** : Documentation AIDD complète
  - [`AIDD_GUIDE.md`](./aidd_docs/AIDD_GUIDE.md) : Guide complet
  - [`UPDATES.md`](./aidd_docs/UPDATES.md) : Mises à jour
  - [`WORKFLOW.md`](./aidd_docs/WORKFLOW.md) : Workflow AIDD
  - [`tasks/`](./aidd_docs/tasks/) : Plans de tâches
  - [`reviews/`](./aidd_docs/reviews/) : Challenges validés
- **[hermes-vault](https://github.com/RebelliousSmile/hermes-vault)** : Compétences custom
- **[aidd-custom](https://github.com/RebelliousSmile/aidd-custom)** : CLI AIDD

## 🔧 Configuration

### **Variables d'environnement**

```bash
# GitHub
GITHUB_TOKEN=your_token

# Together.ai
TOGETHER_API_KEY=your_key

# Fireworks.ai
FIREWORKS_API_KEY=your_key

# Weights & Biases
WANDB_API_KEY=your_key
```

### **Compétences Hermes**

Les compétences sont installées dans `~/.hermes/plugins/hermes-vault/` et liées vers `~/.hermes/skills/`.

Après redémarrage de Hermes Agent, utilisez-les avec :
```bash
hermes -s aidd-workflow "commande"
```

## 🤝 Contribution

1. Fork le dépôt
2. Créer une branche (`git checkout -b feature/awesome`)
3. Commit les changements (`git commit -m 'Add awesome feature'`)
4. Push vers la branche (`git push origin feature/awesome`)
5. Ouvrir une Pull Request

## 📄 Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## 📞 Contact

- **Auteur** : [RebelliousSmile](https://github.com/RebelliousSmile)
- **GitHub** : [suddenly-ai-hub](https://github.com/RebelliousSmile/suddenly-ai-hub)
- **Discord** : #suddenly-ai-hub

---

**Ce projet utilise le framework AIDD (AI-Driven Development) pour structurer le développement de manière professionnelle et systématique.**
