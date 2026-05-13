# suddenly-ai-hub

Fine-tuning de modèles de jeu de rôle (RP) en français

## 🎯 Objectif

Entraîner des modèles LLM (Mistral, Llama) sur des données de RP francophones
pour améliorer leurs compétences en narration, dialogue, et immersion.

## 📁 Structure du projet

```
suddenly-ai-hub/
├── scripts/
│   ├── scrape_couroberon.py      # Scraping La Cour d'Obéron
│   ├── clean_dataset.py           # Nettoyage et anonymisation (à créer)
│   ├── convert_to_axolotl.py      # Conversion format Axolotl (à créer)
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
│   └── README.md                  # Ce fichier
├── tests/                         # Tests unitaires
├── logs/                          # Logs d'exécution
├── requirements.txt               # Dépendances Python
├── .env                           # Variables d'environnement
└── README.md                      # Documentation principale
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
```

### 3. Lancer le scraping
```bash
# Script principal
python scripts/scrape_couroberon.py
```

### 4. Nettoyer les données
```bash
python scripts/clean_dataset.py
```

### 5. Fine-tuning
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

## 🎓 Ressources

- [Guide de scraping](docs/SCRAPING_GUIDE.md)
- [Format des données](docs/DATA_FORMAT.md)
- [Métriques d'évaluation](docs/EVALUATION.md)

## 📝 License

Projet éducatif - données utilisées avec autorisation des propriétaires

## 👥 Contribution

Les contributions sont les bienvenues ! Pour participer :
1. Fork le projet
2. Crée une branche (`git checkout -b feature/ton-feature`)
3. Commit tes changements (`git commit -m 'Ajout feature'`)
4. Push (`git push origin feature/ton-feature`)
5. Ouvre une Pull Request

---

**Dernière mise à jour:** 2026-05-13
