# suddenly-ai-hub

Fine-tuning de modèles de jeu de rôle (RP) en français

## 🎯 Objectif

Entraîner des modèles LLM (Mistral, Mixtral, Llama) sur des données de RP francophones
pour améliorer leurs compétences en narration, dialogue, et immersion.

**Projet en cours de développement** - Scraping de forums JDR + Fine-tuning LoRA

## 📁 Structure du projet

```
suddenly-ai-hub/
├── scripts/
│   ├── scrape_couroberon.py      # ✅ Scraping La Cour d'Obéron
│   ├── clean_dataset.py           # ⏸️ Nettoyage et anonymisation
│   ├── convert_to_axolotl.py      # ⏸️ Conversion format Axolotl
│   ├── test_model.py              # ⏸️ Test du modèle entraîné
│   └── generate_synthetic.py      # ⏸️ Génération de données synthétiques
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
├── .gitignore                     # Patterns exclus
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
TOGETHER_API_KEY=ton_api_key_fireworks_api_key=ton_api_key
GITHUB_TOKEN=ton_github_token
HF_TOKEN=ton_huggingface_token
```

### 3. Lancer le scraping

```bash
# Créer un compte sur La Cour d'Obéron
# https://couroberon.com/Salons/ucp.php?mode=register

# Lancer le scraper avec tes identifiants
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

## 🛠️ Outils & Frameworks

### Scraping
- **BeautifulSoup4** : Parsing HTML
- **Requests** : Requêtes HTTP
- **Playwright** : Automation browser (optionnel)

### Fine-tuning
- **Axolotl** : Configuration YAML pour LoRA/QLoRA
- **Together.ai** : API cloud pour fine-tuning ($30 crédits)
- **Fireworks.ai** : Alternative cloud ($6 crédits)
- **Hugging Face** : Hosting et inference endpoints

### Évaluation
- **lm-eval-harness** : Benchmarks (MMLU, GSM8K, etc.)
- **Human evaluation** : Grille de qualité RP (1-5 étoiles)

## 📝 Formats de données

### JSONL (JSON Lines)

Format standard pour les datasets de fine-tuning :

```json
{"text": "<system>Bonjour, je suis un assistant RP</system>\n<User>Jouerais-tu avec moi ?</User>\n<Model>Avec plaisir ! Quelle histoire veux-tu explorer ?</Model>"}
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

## 📊 Métriques d'évaluation

Après le fine-tuning, évaluer le modèle avec :

- **Qualité narrative** : Score 1-5 sur immersion et cohérence
- **Cohérence des personnages** : Personnalité maintenue
- **Grammaire et style** : Français correct et élégant
- **Creativité** : Réponses originales et intéressantes

## ⚠️ Avertissements légaux

- **Respecte les conditions d'utilisation** des forums scrapés
- Utilise uniquement pour l'entraînement personnel (non commercial)
- Anonymise toujours les données personnelles
- Ne redistribue pas les données originales sans autorisation
- Respecte les délais entre requêtes (3 secondes min.)

## 👥 Contribution

Les contributions sont les bienvenues ! Pour participer :
1. Fork le projet
2. Crée une branche (`git checkout -b feature/ton-feature`)
3. Commit tes changements (`git commit -m 'Ajout feature'`)
4. Push (`git push origin feature/ton-feature`)
5. Ouvre une Pull Request

---

**Dernière mise à jour:** 2026-05-13
**Author:** RebelliousSmile
**License:** MIT (à définir)
