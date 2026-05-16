# Suddenly AI Playground

Petite GUI Gradio pour :
1. **Voir la palette** des 9 features Suddenly et tester chaque feature inline.
2. **Comparer deux modèles en aveugle** (banc A/B) avec vote persisté → utile pour mesurer l'amélioration entre deux runs de fine-tuning.
3. **Vérifier l'état du gateway** `muse.suddenly.social` et la couverture de la palette (quels adapters sont effectivement servis).

## Installation

```bash
pip install -e ".[playground]"
```

## Configuration

Variables d'environnement :

| Variable | Rôle | Défaut |
|---|---|---|
| `SUPPORTING_PROVIDER` | `together` ou `fireworks` (auto-détection sinon) | — |
| `TOGETHER_API_KEY` | clé Together.ai | — |
| `FIREWORKS_API_KEY` | clé Fireworks.ai | — |
| `SUDDENLY_HUB_URL` | URL du gateway Suddenly | `https://muse.suddenly.social` |

## Lancement

```bash
python -m apps.playground.app
```

Ouvre `http://127.0.0.1:7860`.

## Banc A/B : protocole

1. Fixe un set de prompts de référence (ceux de `pipelines/evaluation/test-prompts/*.jsonl` font l'affaire).
2. Pour chaque prompt, génère la comparaison entre la baseline (ex. `Qwen/Qwen2.5-7B-Instruct`) et ton run de fine-tuning courant.
3. Vote ◀/=/▶ en aveugle (l'ordre gauche/droite est randomisé).
4. Les votes sont append dans `data/playground/votes.csv` (gitignored).
5. Le scoreboard agrège wins/losses/ties par modèle pour visualiser la progression.

## Limitation actuelle

L'endpoint `/v1/chat/completions` du gateway exige une signature HTTP ActivityPub. Le playground ne signe pas les requêtes — il passe directement par les providers Together/Fireworks où les LoRAs sont (ou seront) hébergés. L'onglet **Gateway** ne fait que des appels publics (`/v1/models`, `/v1/health`, `/v1/stats`).
