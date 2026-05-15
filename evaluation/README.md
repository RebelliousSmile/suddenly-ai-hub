# Évaluation des LoRA Suddenly

Suite de tests pour valider la qualité des modèles de rôle RP français spécialisés par LoRA.

## Structure

```
evaluation/
├── spec-tests-lora.md          # Specification complète du plan de tests
├── evaluate_lora.py            # Script principal d'évaluation
├── test-prompts/
│   ├── genres.jsonl            # 14 prompts par genre
│   ├── situations.jsonl        # 6 prompts par situation
│   ├── couples.jsonl           # 6 couples genre+situation
│   └── generic.jsonl           # 1 prompt générique
├── reports/
│   ├── results.csv             # Résultats bruts
│   └── results.json            # Résultats détaillés + résumé
└── README.md                   # Ce fichier
```

## Prompt generation

27 prompts au total :

- **14 genres** : médiéval fantastique, sci-fi, cyberpunk, etc.
- **6 situations** : combat, romance, enquête, diplomatie, exploration, introspection
- **6 couples** : combinaisons genre+situation critiques
- **1 générique** : fallback neutre

## Utilisation

### Mode simulation (par défaut, sans modèle)

```bash
python evaluate_lora.py --mode simulate
```

Génère des réponses fictives pour tester le pipeline d'évaluation.

### Mode local (vLLM)

```bash
# Lancer un endpoint vLLM local
vllm serve /path/to/suddenly-7b-lora --port 8000

# Évaluer
python evaluate_lora.py --mode local --model suddenly-7b-lora
```

### Mode API (Together.ai)

```bash
export TOGETHER_API_KEY="your-key"
python evaluate_lora.py --mode api --provider together --model suddenly-7b-lora --api-key $TOGETHER_API_KEY
```

### Mode API (Fireworks.ai)

```bash
export FIREWORKS_API_KEY="your-key"
python evaluate_lora.py --mode api --provider fireworks --model accounts/fireworks/models/suddenly-7b-lora --api-key $FIREWORKS_API_KEY
```

## Critères d'évaluation

Chaque réponse est notée sur 5 critères (1-5) :

| Critère | Description |
|---------|-------------|
| Cohérence thématique | Le modèle reste-t-il dans le genre/situation ? |
| Créativité | Les réponses sont-elles originales ? |
| Profondeur émotionnelle | Les émotions sont-elles adaptées ? |
| Style | Le ton correspond-il au genre/situation ? |
| Immersion | La réponse est-elle immersive et bien rédigée ? |

Score global = moyenne des 5 critères × 20 (échelle 0-100).

## Livrables attendus

- `reports/results.csv` — scores par test
- `reports/results.json` — détails complets + résumé par catégorie
- Résumé affiché en console avec top/bottom des tests

## Améliorations futures

- [ ] Mode `eval-llm` : utiliser un LLM évaluateur (GPT-4/Claude) pour noter les réponses
- [ ] Support du mode pre-merge (tester les couples fusionnés)
- [ ] Benchmark comparatif modèle de base vs LoRA
- [ ] Dashboard web pour visualiser les résultats
