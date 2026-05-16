# 🏗️ Architecture du projet Suddenly AI Hub

Ce document décrit l'architecture technique du projet, la chaîne de fabrication des modèles fine-tunés, et les décisions techniques prises.

## Contexte

Les modèles sont fine-tunés pour les **features AI de Suddenly** (#76-#84) :

1. **#77** — Suggestion de dialogue
2. **#78** — Suggestion d'action
3. **#79** — Suggestion de description de scène
4. **#80** — Suggestion de pensée intérieure
5. **#81** — Analyse de cohérence RP (scène)
6. **#82** — Analyse de cohérence RP (session)
7. **#83** — Résumé de session
8. **#84** — Suggestions de liens fédérés

Chaque feature correspond à un LoRA adapter spécifique.

## Stack technique

- **Base model** : Qwen2.5-7B-Instruct
- **Framework training** : Axolotl (via Together.ai)
- **Optimization** : LoRA (Low-Rank Adaptation)
- **Environnement** : AIDD + Hermes Agent

## Structure du projet

```
suddenly-ai-hub/
├── models/                    # LoRA adapters fine-tunés
│   ├── suddenly-dialogue/     # #77
│   ├── suddenly-action/       # #78
│   ├── suddenly-description/  # #79
│   ├── suddenly-thought/      # #80
│   ├── suddenly-consistency-scene/   # #81
│   ├── suddenly-consistency-session/  # #82
│   ├── suddenly-summary/      # #83
│   └── suddenly-federation/   # #84
├── scripts/
│   ├── train_together.py      # Pipeline training via Together.ai
│   ├── list_models.py         # Registry des adapters
│   ├── infer.py               # CLI inference
│   └── utils/
├── aidd_docs/
│   └── memory/
│       └── architecture.md    # Ce fichier
└── README.md                  # Documentation utilisateur (usage uniquement)
```

## Pipeline de données

5 sources de données RP français :

1. **Discord RP logs** — RPDiscord, RPDiscord2, RPDiscord3 (formats variés)
2. **Forum RP** — La Cour d'Obéron, autres forums (formats variés)
3. **Ren'Py dialogues** — extraits de dialogues de jeux textuels français
4. **Google Books** — extraits de fiction RP française
5. **Playwright scraping** — contenu RP (via Playwright)

## Format Axolotl

Les données sont converties au format Axolotl :

```json
{
  "conversations": [
    {"from": "human", "value": "Prompt utilisateur"},
    {"from": "gpt", "value": "Réponse RP"}
  ]
}
```

## Configuration d'entraînement

```yaml
# train_together.py
base_model: "Qwen/Qwen2.5-7B-Instruct"
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
epochs: 3
batch_size: 4
learning_rate: 2e-4
```

## Coûts

- **Together.ai** : ~$0.02/heure d'entraînement
- **Hetzner S3** : ~$5/mo pour le stockage des données
- **AIDD** : infrastructure gratuite

## Sécurité

- Les données RP sont anonymisées (noms remplacés par {{char}}/{{user}})
- Les tokens API sont stockés dans les secrets AIDD
- Les modèles fine-tunés sont stockés dans `models/` (git-lfs)

## Évolutions futures

- [ ] Support de Qwen3
- [ ] Fine-tuning complet (pas que LoRA)
- [ ] Dataset crowdsourced (RP community)
- [ ] Benchmarks de qualité RP
