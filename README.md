# 🚀 Suddenly AI Hub

> Fine-tuning d'un modèle de langage JDR (Jeux de Rôle) en français avec le framework AIDD

## 📋 Résumé du Projet

Ce projet vise à fine-tuner un modèle de langage pour la génération de campagnes de jeu de rôle en français. Utilisant le framework **AIDD** (AI-Driven Development) pour structurer le développement.

### 🎯 Objectifs

- **Phase 1-2:** Scraping de données jdRoll (campagnes JDR)
- **Phase 3:** Fine-tuning du modèle JDR
- **Phase 4:** Amélioration continue
- **Phase 5:** Scaling vers 100+ campagnes
- **Phase 6:** Production
- **Phase 7:** Communauté

## 🚀 Démarrage Rapide

### 1️⃣ Installer le CLI AIDD

```bash
# Installer aidd-custom
npm install -g aidd-custom

# Configurer le framework
cd /home/user/suddenly-ai-hub
aidd-custom setup --repo RebelliousSmile/aidd-overlay

# Installer pour Copilot (recommandé)
aidd-custom install --ai copilot
```

### 2️⃣ Vérifier l'Installation

```bash
aidd-custom doctor
aidd-custom status
```

### 3️⃣ Lire la Documentation

- 📖 **[INSTALL.md](./INSTALL.md)** - Installation rapide
- 📖 **[aidd_docs/AIDD_GUIDE.md](./aidd_docs/AIDD_GUIDE.md)** - Guide complet AIDD
- 📖 **[aidd_docs/WORKFLOW.md](./aidd_docs/WORKFLOW.md)** - Workflow détaillé

## 📁 Structure du Projet

```
suddenly-ai-hub/
├── aidd_docs/                    # Documentation AIDD
│   ├── tasks/                    # Plans de phases
│   │   ├── phase3_finetuning.md  # Fine-tuning
│   │   ├── phase4_improvement.md # Amélioration
│   │   ├── phase5_scaling.md     # Scaling
│   │   └── phase6_production.md  # Production
│   ├── reviews/                  # Challenges validés
│   ├── memory/                   # Mémoire (décisions, lessons)
│   ├── changelog/                # CHANGELOG.md
│   └── AIDD_GUIDE.md             # Guide complet
├── scripts/                      # Scripts Python
│   ├── session2_simple.py        # Scraping jdRoll
│   ├── convert_to_jsonl.py       # Conversion données
│   └── train_model.py            # Fine-tuning
├── .env                          # Variables d'environnement (NE PAS commit)
├── README.md                     # Ce fichier
└── INSTALL.md                    # Installation rapide
```

## 🔧 Configuration

### Variables d'Environnement

Créer `.env` dans le dossier racine :

```bash
# GitHub (pour les PR et le CLI)
GITHUB_TOKEN=ghp_********

# Together.ai (pour l'API)
TOGETHER_API_KEY=sk-********

# Fireworks.ai (pour l'API)
FIREWORKS_API_KEY=sk-********

# HuggingFace (pour les modèles)
HF_TOKEN=hf_********
```

### Clés API et Sécurité

❌ **NE JAMAIS** commit de `.env`  
✅ **Toujours** utiliser `~/.hermes/.env` pour les clés  
✅ **Toujours** utiliser des tokens avec permissions minimales

## 📚 Documentation

### Guides d'Installation

- **[INSTALL.md](./INSTALL.md)** - Installation en 5 minutes
- **[aidd_docs/AIDD_GUIDE.md](./aidd_docs/AIDD_GUIDE.md)** - Guide complet AIDD

### Documentation AIDD

- **[aidd_docs/WORKFLOW.md](./aidd_docs/WORKFLOW.md)** - Workflow complet
- **[aidd_docs/tasks/phase3_finetuning.md](./aidd_docs/tasks/phase3_finetuning.md)** - Plan Phase 3
- **[aidd_docs/reviews/challenge_phase3.md](./aidd_docs/reviews/challenge_phase3.md)** - Challenge Phase 3

### Ressources Externes

- **AIDD CLI:** https://github.com/RebelliousSmile/aidd-custom
- **AIDD Overlay:** https://github.com/RebelliousSmile/aidd-overlay
- **Hermes Agent:** https://hermes-agent.nousresearch.com/

## 🎯 Phases du Projet

### ✅ Phase 1-2: Scraping (Complet)

- ✅ Scraping de jdRoll (20 campagnes)
- ✅ Extraction des données (title, universe, system, author, description)
- ✅ Préparation des données pour le fine-tuning

### 🔄 Phase 3: Fine-tuning (En cours)

- 🔄 Conversion des données en JSONL
- 🔄 Fine-tuning avec Axolotl/Unsloth
- 🔄 Évaluation du modèle

### ⏭️ Phase 4: Amélioration Continue

- 📊 Collecte de feedback
- 🔧 Amélioration des prompts
- 📈 Optimisation des performances

### ⏭️ Phase 5: Scaling

- 📦 Augmentation des données (100+ campagnes)
- 🚀 Optimisation de l'infrastructure
- 📊 Monitoring des performances

### ⏭️ Phase 6: Production

- 🎯 Déploiement en production
- 🔐 Sécurisation des accès
- 📈 Monitoring et alerting

### ⏭️ Phase 7: Communauté

- 🌐 Publication des modèles
- 📚 Documentation communautaire
- 💬 Support et feedback

## 🛠 Utilisation du Framework AIDD

### Commandes Essentielles

```bash
# Configuration
aidd-custom setup --repo RebelliousSmile/aidd-overlay

# Installation
aidd-custom install --ai copilot

# Maintenance
aidd-custom doctor
aidd-custom update
aidd-custom status
```

### Workflow Typique

```bash
# 1. Créer un plan
aidd-custom plan create --name "feature-x" --issue 50

# 2. Challenger le plan
aidd-custom plan challenge phase3_finetuning.md

# 3. Implémenter

# 4. Documenter
aidd-custom learn --phase 3

# 5. Mettre à jour le changelog
vim aidd_docs/changelog/CHANGELOG.md
```

## 📊 Statistiques

| Élément | Statut | Détails |
|---------|--------|---------|
| **Campaignes scrapées** | ✅ 20 | jdRoll |
| **Plans AIDD** | ✅ 5 | Phase 3-7 |
| **Challenges** | ✅ 5 | Validés |
| **Scripts Python** | ✅ 3 | Scraping + training |
| **Documentation AIDD** | ✅ 44+ fichiers | Structure complète |

## 🤝 Contribution

1. Fork le projet
2. Créer une branche feature (`git checkout -b feature/amazing`)
3. Commit les changements (`git commit -m 'Add amazing feature'`)
4. Push vers la branche (`git push origin feature/amazing`)
5. Ouvrir une Pull Request

## 📝 License

Ce projet est soumis à la license MIT.

## 📞 Support

- **GitHub Issues:** https://github.com/RebelliousSmile/suddenly-ai-hub/issues
- **Documentation:** `aidd_docs/`

---

**Projet maintenu avec ❤️ par RebelliousSmile**

*Dernière mise à jour: 2026-05-13*
*Version: 1.0.0*
