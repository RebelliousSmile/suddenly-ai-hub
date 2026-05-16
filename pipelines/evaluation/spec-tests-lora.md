# Spec — Plan de tests LoRA Suddenly

> **Issue** : #57 — Créer les tests de validation LoRA par genre et situation
> **Date** : 2026-05-15

---

## 1. Objectif

Valider que chaque LoRA Suddenly produit des réponses cohérentes avec le genre/situation visé, avant déploiement en production.

---

## 2. Structure des tests

### 2.1. Prompts par genre (14 tests)
Chaque genre reçoit un prompt unique testant la cohérence thématique.

| Genre | Prompt type |
|-------|-------------|
| Médiéval Fantastique | Quête de chevalier avec magie |
| Historique Fantastique | Période historique + élément surnaturel |
| Science Fiction | Futuriste, technologie avancée |
| Contemporain Fantastique | Monde moderne + surnaturel |
| Space Opera | Conflit interstellaire |
| Contemporain | Monde réel moderne |
| Post-Apocalyptique | Survie après catastrophe |
| Cyberpunk / Anticipation | Futur proche, sombre, technologique |
| Super-héros | Pouvoirs, justice, organisations |
| Oriental / Manga | Thèmes asiatiques, style manga |
| Merveilleux / Onirique | Rêve, magie, absurde |
| Humoristique | Comédie, parodie, léger |
| Univers parallèles | Multivers, réalités alternatives |
| Générique | Fallback neutre, aucune thématique spécifique |

### 2.2. Prompts par situation (6 tests)
Chaque situation reçoit un prompt testant le style narratif.

| Situation | Prompt type |
|-----------|-------------|
| Romance / Relation | Interaction émotionnelle, séduction |
| Combat / Bataille | Action physique, stratégie, violence |
| Enquête / Investigation | Mystère, indices, raisonnements |
| Diplomatie / Négociation | Politiques, échanges verbaux |
| Exploration / Découverte | Découverte de lieux, artefacts |
| Introspection / Monologue | Réflexion intérieure, sentiments profonds |

### 2.3. Couples critiques (6 tests)
Combinaisons genre + situation pour valider la spécialisation combinée.

| Couple | Description |
|--------|-------------|
| Medieval-Fantastique + Combat | Quête épique avec bataille |
| Cyberpunk + Enquête | Détective dans un monde sombre |
| Contemporain + Romance | Histoire d'amour moderne |
| Science-Fiction + Exploration | Découverte de planète inconnue |
| Post-Apocalyptique + Introspection | Survivant réfléchissant |
| Space-Opera + Diplomatie | Négociation interstellaire |

### 2.4. Test générique (1 test)
Prompt neutre pour valider le LoRA générique (fallback).

---

## 3. Critères d'évaluation

Chaque réponse est notée sur une échelle 1-5 pour chaque critère :

| Critère | Description | Score 1 | Score 5 |
|---------|-------------|---------|---------|
| **Cohérence thématique** | Le modèle reste-t-il dans le genre/situation ? | Hors sujet | Parfaitement aligné |
| **Créativité** | Les réponses sont-elles originales ? | Clichés | Surprenant et inventif |
| **Profondeur émotionnelle** | Les émotions sont-elles adaptées ? | Plat | Touchant et nuancé |
| **Style** | Le ton correspond-il au genre/situation ? | Inadapté | Parfaitement adapté |
| **Immersion** | La réponse est-elle immersive ? | Sèche | Envoûtante et vivante |

---

## 4. Structure des fichiers

```
evaluation/
├── test-prompts/
│   ├── genres.jsonl          # 14 prompts par genre
│   ├── situations.jsonl      # 6 prompts par situation
│   ├── couples.jsonl         # 6 couples critiques
│   └── generic.jsonl         # 1 prompt générique
├── evaluate_lora.py          # Script principal d'évaluation
├── reports/
│   └── results.csv           # Résultats des tests
└── README.md                 # Documentation d'utilisation
```

---

## 5. Format des prompts JSONL

```json
{"id": "medieval-fantastique-combat", "category": "genre", "name": "Médiéval Fantastique", "prompt": "...", "expected_style": "épopée, magie, chevalerie"}
```

---

## 6. Sortie du script

Le script `evaluate_lora.py` génère :
- Un rapport CSV avec scores par critère
- Un rapport JSON avec détails des réponses
- Un résumé statique (score moyen, meilleur genre, pire genre)

---

## 7. Modes d'exécution

### Mode local (vLLM)
Le script envoie les prompts à un endpoint vLLM local.

### Mode API
Le script envoie les prompts à une API externe (Together.ai, Fireworks.ai).

### Mode simulation
Le script génère des réponses fictives pour tester le pipeline d'évaluation sans modèle.
