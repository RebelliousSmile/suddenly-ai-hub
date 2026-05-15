# 🏗️ Architecture du projet Suddenly AI Hub

Ce document décrit en détail l'architecture technique du projet, la chaîne de fabrication des modèles fine-tunés, et les décisions techniques prises.

## Objectif du projet

Fine-tuner des modèles de langage pour générer des dialogues et narrations JDR en français, spécialisé dans différents univers (Cyberpunk, Fantasy, Horreur, etc.).

## Stack technique

| Composant | Outil | Rôle |
|-----------|-------|------|
| **Framework LLM** | Together.ai | API cloud pour fine-tuning et inférence |
| **Modèle de base** | Qwen/Qwen2.5-7B-Instruct | Point de départ pour le fine-tuning SFT |
| **Format d'entraînement** | Axolotl JSONL | Format messages standardisé |
| **Collection de données** | GitHub API + parsing Ren'Py | Recherche et extraction de dialogues VN |
| **Scraping web** | Playwright | Sites modernes JS-heavy |
| **Tracking** | WandB | Suivi des expériences et métriques |

## Structure du projet

```
suddenly-ai-hub/
├── scripts/                      # Outils de fabrication
│   ├── train_together.py         # Client principal Together.ai
│   │   ├── validate              # Validation JSONL + split train/val
│   │   ├── train                 # Upload + fine-tuning job
│   │   └── infer                 # Inférence sur val avec modèle fine-tuned
│   │
│   ├── crawl_rpv/                # Pipeline Ren'Py
│   │   ├── generate_corpus.py    # Générateur corpus synthétique (5 genres)
│   │   ├── github_search.py      # Recherche VNs sur GitHub (auto-token)
│   │   ├── extract_real_dialogues.py # Extraction dialogues depuis .rpy locaux
│   │   ├── scan_dialogue_files.py    # Scanner .py/.rpy détection dialogues
│   │   ├── rpv_pipeline.py       # Pipeline unifié orchestration
│   │   └── select_dialogue_files.py  # Score qualité fichiers dialogue
│   │
│   ├── scrape_couroberon.py      # Scraping forum RP français
│   ├── scrape_jdroll.py          # Scraping JD Roll
│   └── setup-python-env.sh       # Environnement Python
│
├── data/                         # Données brutes et entraînement
│   ├── renpy-corpus.jsonl        # Corpus synthétique français (5 genres, ~50 entries)
│   ├── renpy-corpus-test.jsonl   # Test set
│   ├── renpy-real-dialogues.jsonl # Dialogues extraits de VNs GitHub
│   ├── renpy-repos.json          # Repos GitHub trouvés
│   ├── train.jsonl               # Dataset entraînement (Axolotl format)
│   └── val.jsonl                 # Dataset validation
│
├── models/                       # Modèles entraînés et checkpoints
│   └── (stocké localement ou externe)
│
├── .cache/                       # Cache local
│   └── model_id.txt              # ID du dernier modèle entraîné
│
└── README.md                     # Documentation utilisateur publique
```

## Pipeline de données

### 1. Collecte multi-source

| Source | Type | Langue | Méthode | Statut |
|--------|------|--------|---------|--------|
| **Corpus synthétique Ren'Py** | Scènes + dialogues | Français | Générateur 5 genres | ✅ Actif |
| **VNs GitHub** | Dialogues réels | Anglais | GitHub API + parsing .rpy | ✅ Actif |
| **Forum RP La Cour d'Obéron** | Roleplay forum | Français | Playwright scraping | 🚧 En attente |
| **JD Roll** | Forum JDR | Français | Playwright scraping | 🚧 En attente |
| **Google Books API** | Romans public domain | Français | REST API | 🚧 En attente |

### 2. Corpus synthétique Ren'Py (priorité)

**Générateur** `scripts/crawl_rpv/generate_corpus.py`
- 5 genres : contemporain, fantaisie médiévale, horreur surnaturelle, science-fiction, seinen drama
- ~10 entries par genre
- Format Axolotl messages : dialogue + contexte de scène
- Sortie : `data/renpy-corpus.jsonl`

### 3. Extraction dialogues VN GitHub

**Recherche** `scripts/crawl_rpv/github_search.py`
- Auto-détection token `gh` via `gh auth token`
- Recherche `renpy stars:>20` → 170+ repos
- ⚠️ **Attention** : `?recursive=1` obligatoire sur l'API tree GitHub pour scanner les sous-répertoires

**Extraction** `scripts/crawl_rpv/extract_real_dialogues.py`
- Parsing fichiers `.rpy` (Ren'Py)
- Détection dialogue patterns : `"text"`, dialogue variables, labels
- Format Axolotl pour intégration

**Limitation** : VNs GitHub sont principalement en anglais. Stratégie : traduction EN→FR comme étape suivante.

## Pipeline d'entraînement

### Configuration Together.ai

**Script principal** : `scripts/train_together.py`

```bash
# Validation du dataset
python scripts/train_together.py validate --input data/train.jsonl

# Entrainement
python scripts/train_together.py train \
  --model Qwen/Qwen2.5-7B-Instruct \
  --lr 0.0001 \
  --epochs 3

# Inférence
python scripts/train_together.py infer \
  --model-id {latest_model_id} \
  --system-prompt "Tu es un conteur..." \
  --messages '[{"role":"user","content":"..."}]'
```

**Modèles fine-tunables disponibles** :
- `Qwen/Qwen2.5-7B-Instruct` (principal)
- `Qwen/Qwen2.5-14B-Instruct` (alternative)
- `meta-llama/Llama-3.1-70B-Instruct-Turbo` (option)

### Format de données Axolotl

Chaque ligne est un JSON avec une liste de messages :

```json
{
  "messages": [
    {"role": "system", "content": "Tu es un PNJ dans un cyberpunk..."},
    {"role": "user", "content": "Bonjour, je cherche du travail."},
    {"role": "assistant", "content": "Y'a la taverne en bas du quartier, mais y'a pas d'emploi, hein..."}
  ]
}
```

### Split des données

```
data/train.jsonl   → 80% données (entraînment)
data/val.jsonl     → 20% données (validation)
```

## Stratégies d'usage des modèles

### 1. Pattern Maître de Jeu (DM)

```
Tu es un maître de jeu. Tu décris les scènes, incarnes les PNJ et gères l'action.
Le joueur dit: "{user_input}"
```

### 2. Pattern PNJ

```
Tu es un marchand cynique dans une taverne cyberpunk. Tu vends des informations, pas des armes.
Le joueur dit: "{user_input}"
```

### 3. Pattern Narrateur

```
Tu racontes une scène de jeu. Style sombre, descriptions sensorielles, rythme tendu.
Le joueur dit: "{user_input}"
```

## Infrastructure

| Composant | Hébergement | Coût |
|-----------|-------------|------|
| **Code** | GitHub (RebelliousSmile/suddenly-ai-hub) | Gratuit (MIT) |
| **Fine-tuning** | Together.ai (API cloud) | Payant (usage) |
| **Tracking** | WandB | Gratuit (public) |
| **Stockage** | Local + stockage externe | Variable |
| **Container** | Docker (nikolaik/python-nodejs) | Gratuit |

## Sécurité

- **Aucune donnée sensible** dans le code (tokens, API keys)
- Variables d'environnement pour les credentials
- `.aidd/hooks/pre-commit` : vestige Docker, pas utilisé (ignorer warning)
- Licence MIT : utilisation commerciale autorisée

## Évolutions prévues

| Feature | Statut | Priorité |
|---------|--------|----------|
| Traduction EN→FR des dialogues VN | 🚧 Planifié | Haute |
| Scraping forum RP français | 🚧 En attente | Moyenne |
| Support multi-modèles (14B, 70B) | 📋 Planifié | Moyenne |
| Évaluation automatique des sorties | 📋 Planifié | Moyenne |
| Intégration API REST pour inférence | 📋 Planifié | Basse |

## Métriques trackées

- Loss d'entraînement (WandB)
- Qualité des sorties (évaluations manuelles)
- Latence inférence
- Coût par génération

---

**Ce document est interne au projet. Ne pas diffuser publiquement.**
