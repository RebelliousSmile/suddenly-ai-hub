# ============================================
# Suddenly AI Hub - Documentation
# ============================================

## 🎯 Objectif

Entraîner des modèles LLM (Qwen2.5, Llama) sur des données de RP francophones
pour améliorer leurs compétences en narration, dialogue, et immersion.

Ce projet utilise **aidd-framework** (via `aidd-custom` et `aidd-overlay`)
pour structurer le code et gérer les outils de développement.

## 📁 Structure du projet

```
suddenly-ai-hub/
├── scripts/
│   ├── scrape_couroberon.py      # Scraping La Cour d'Obéron
│   ├── clean_dataset.py           # Nettoyage et anonymisation
│   ├── convert_to_axolotl.py      # Conversion format Axolotl
│   ├── test_model.py              # Test du modèle entraîné
│   └── generate_synthetic.py      # Génération de données synthétiques
├── data/
│   ├── raw/                       # Données brutes scrapées
│   ├── clean/                     # Données nettoyées et anonymisées
│   └── eval/                      # Dataset d'évaluation (100 exemples)
├── config/
│   ├── axolotl_config.yaml        # Configuration Axolotl
│   ├── together_config.yaml       # Configuration Together.ai
│   └── fireworks_config.yaml      # Configuration Fireworks.ai
├── docs/
│   ├── SCRAPING_GUIDE.md          # Guide de scraping
│   ├── DATA_FORMAT.md             # Format des données
│   ├── EVALUATION.md              # Métriques d'évaluation
│   └── AIDD_INTEGRATION.md        # Guide d'intégration aidd
├── .gitignore                     # Patterns exclus (issues de aidd-custom)
├── requirements.txt               # Dépendances Python
├── .env                           # Variables d'environnement (API keys)
└── README.md                      # Ce fichier
```

## 🚀 Démarrage rapide

### 1. Installer les dépendances

```bash
cd /home/user/suddenly-ai-hub
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configurer les API keys

```bash
# Créer .env avec tes clés
echo "TOGETHER_API_KEY=ton_api_key" >> .env
echo "FIREWORKS_API_KEY=ton_api_key" >> .env
echo "GITHUB_TOKEN=ton_github_token" >> .env
echo "HF_TOKEN=ton_huggingface_token" >> .env
```

### 3. Intégrer aidd-framework

```bash
# Option A: Comme sous-module (recommandé)
git clone https://github.com/RebelliousSmile/aidd-custom.git ../aidd-custom
git clone https://github.com/RebelliousSmile/aidd-overlay.git ../aidd-overlay

# Option B: Copier localement
cp -r ../aidd-custom/. ./aidd-custom/
cp -r ../aidd-overlay/. ./aidd-overlay/
```

### 4. Lancer le scraping

```bash
# Script principal
python scripts/scrape_couroberon.py
```

### 5. Nettoyer les données

```bash
python scripts/clean_dataset.py
```

### 6. Fine-tuning

```bash
# Avec Fireworks.ai (recommandé pour débuter)
fireworks-cli train --dataset data/clean/dataset.jsonl --model llama-v3-8b-instruct

# Ou avec Axolotl
accelerate launch axolotl/train.py --config config/axolotl_config.yaml
```

## 📋 Issues en cours

Voir: https://github.com/RebelliousSmile/suddenly-ai-hub/issues

- #44 ✅ Préparer l'environnement Python
- #45 🔄 Tester l'API Fireworks.ai
- #46 ⏸️ Créer un dataset test minimal
- #47 ✅ Scraper La Cour d'Obéron
- #48 ⏸️ Nettoyage et anonymisation
- #49 ⏸️ Convertir en format JSONL Axolotl

## 🛠️ Outils & Frameworks

### aidd-framework

Ce projet utilise **aidd-custom** et **aidd-overlay** pour :
- Structurer le code selon des conventions établies
- Gérer les outils de développement
- Fournir des CLI personnalisées
- Standardiser les workflows

**Documentation aidd** : Voir le repo [aidd-custom](https://github.com/RebelliousSmile/aidd-custom)

### Autres outils

- **Axolotl** : Fine-tuning de LLM avec YAML
- **Together.ai** : API pour le fine-tuning cloud
- **Fireworks.ai** : Alternative cloud pour le fine-tuning
- **Hugging Face** : Hosting et inference endpoints
- **BeautifulSoup** : Scraping web
- **Playwright** : Automation browser

## 📝 Formats de données

### JSONL (JSON Lines)

Format standard pour les datasets de fine-tuning :

```json
{
  "text": "<system>Bonjour, je suis un assistant RP</system>\n<User>Jouerais-tu avec moi ?</User>\n<Model>Avec plaisir ! Quelle histoire veux-tu explorer ?</Model>"
}
```

### Format Axolotl

Structure de conversation :

```json
{
  "messages": [
    {"role": "system", "content": "Tu es un narrateur de RP..."},
    {"role": "user", "content": "J'aimerais commencer..."},
    {"role": "assistant", "content": "Très bien !"}
  ]
}
```

## 🎓 Ressources

- [Guide de scraping](docs/SCRAPING_GUIDE.md)
- [Format des données](docs/DATA_FORMAT.md)
- [Métriques d'évaluation](docs/EVALUATION.md)
- [Intégration aidd](docs/AIDD_INTEGRATION.md)

## 📊 Métriques d'évaluation

Après le fine-tuning, évaluer le modèle avec :

- **Qualité narrative** : Score 1-5 sur immersion et cohérence
- **Cohérence des personnages** : Personnalité maintenue
- **Grammaire et style** : Français correct et élégant
- **Creativité** : Réponses originales et intéressantes

## ⚠️ Avertissements légaux

- Respecte les conditions d'utilisation des forums scrapés
- Utilise uniquement pour l'entraînement personnel (non commercial)
- Anonymise toujours les données personnelles
- Ne redistribue pas les données originales sans autorisation

## 👥 Contribution

Les contributions sont les bienvenues ! Pour participer :
1. Fork le projet
2. Crée une branche (`git checkout -b feature/ton-feature`)
3. Commit tes changements (`git commit -m 'Ajout feature'`)
4. Push (`git push origin feature/ton-feature`)
5. Ouvre une Pull Request

## 📞 Support

Pour les problèmes :
- Consulte le README et les docs
- Vérifie les issues GitHub
- Contacte l'équipe (si open source)

---

**Dernière mise à jour:** 2026-05-13
**Author:** RebelliousSmile
**License:** MIT (à définir)
