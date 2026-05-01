# Métriques d'évaluation modèle

**Issue** : #13 | **Date** : 2026-05-01

---

## Contexte

Le dataset `training/eval-dataset.jsonl` contient 100 sessions RP de référence réparties sur les 11 genres GROG. Il est **exclusivement réservé à l'évaluation** — il ne doit jamais apparaître dans les `datasets:` d'un fichier yml Axolotl.

L'évaluation se fait en deux temps :
1. **Avant fine-tuning** : baseline du modèle de base (Mistral 7B v0.3 ou Mistral NeMo 12B)
2. **Après fine-tuning** : modèle suddenly-7b ou suddenly-13b

Le delta entre les deux mesures quantifie l'apport du fine-tuning.

---

## Grille d'évaluation humaine

Notation sur **5 critères**, chacun noté de 1 à 5.

| Critère | 1 (Insuffisant) | 3 (Acceptable) | 5 (Excellent) |
|---|---|---|---|
| **Cohérence narrative** | Contradiction avec le contexte établi | Cohérent sans être mémorable | Personnages et événements cohérents et nuancés |
| **Fluidité stylistique** | Registre générique, formules robotiques | Registre narratif présent mais inconsistant | Prose immersive, rythme maîtrisé, vocabulaire adapté |
| **Pertinence RP** | Ignore l'action du joueur | Répond à l'action sans la développer | Fait avancer l'histoire, ouvre des pistes narratives |
| **Immersion sensorielle** | Aucune description sensorielle | Quelques détails descriptifs | Richesse sensorielle (vue, son, odeur, toucher) |
| **Respect du genre** | Aucun marqueur stylistique du genre | Genre reconnaissable mais peu ancré | Marqueurs forts du genre (archaïsmes, jargon tech, etc.) |

**Score global** : somme des 5 critères / 25 → normalisé sur 100.

**Seuils recommandés** :
- Score < 40/100 : qualité insuffisante pour production
- Score 40–60/100 : qualité acceptable, améliorations souhaitables
- Score > 60/100 : qualité production

### Protocole d'annotation humaine

1. Sélectionner 20 exemples aléatoires dans le dataset d'éval (2 par genre)
2. Générer les réponses du modèle sur ces 20 exemples
3. Faire annoter par 2 annotateurs indépendants
4. Calculer l'accord inter-annotateurs (Cohen's Kappa cible : > 0.6)
5. En cas de désaccord > 1 point, arbitrage par un troisième annotateur

---

## Métriques automatiques

### chrF++ (sacrebleu)

**Principe** : F-mesure sur n-grammes de caractères et de mots entre la réponse du modèle et la réponse de référence. Plus robuste que BLEU pour le français morphologiquement riche.

**Seuils** :
| Score | Interprétation |
|---|---|
| < 0.15 | Très faible — réponse hors sujet |
| 0.15–0.25 | Faible — réponse partiellement pertinente |
| 0.25–0.40 | Acceptable — réponse dans le bon registre |
| ≥ 0.40 | Bon — réponse proche de la référence en style et contenu |

> **Note** : chrF mesure la similitude avec la référence, pas la qualité absolue. Une réponse très créative mais stylistiquement différente de la référence peut avoir un score faible tout en étant bonne — la compléter avec l'évaluation humaine.

### BERTScore (optionnel)

**Principe** : similitude sémantique entre réponse et référence via embeddings contextuels. Modèle utilisé : `camembert-base` (FR). Nécessite `torch` et `bert-score`.

**Seuils** :
| F1 | Interprétation |
|---|---|
| < 0.75 | Très faible — peu de chevauchement sémantique |
| 0.75–0.80 | Faible |
| 0.80–0.88 | Acceptable |
| ≥ 0.88 | Bon |

### Ratio longueur

Longueur de la réponse du modèle / longueur de la réponse de référence.

| Ratio | Interprétation |
|---|---|
| < 0.3 | Réponse trop courte (troncature probable) |
| 0.3–0.5 | Réponse courte |
| 0.5–2.0 | **Acceptable** |
| > 2.0 | Réponse trop longue (verbosité) |

### Taux de répétition

Proportion de trigrammes répétés dans la réponse. Un modèle en boucle répétitive a un taux élevé.

| Taux | Interprétation |
|---|---|
| < 0.03 | Excellent — pas de répétition |
| 0.03–0.08 | Acceptable |
| > 0.08 | Problématique — réponse répétitive |

### % Réponses en français

Proportion de réponses détectées en français par `langdetect`.

**Cible** : 100%. Une valeur < 95% indique un problème de langue (réponse en anglais ou mixte).

---

## Utilisation du script evaluate.py

```bash
# 1. Générer les prédictions du modèle
#    (utiliser vLLM ou Hugging Face transformers pour inférence)
#    Format attendu : même que eval-dataset.jsonl
#    mais avec les contenus assistant remplacés par les sorties du modèle

# 2. Lancer l'évaluation
python pipeline/evaluate.py \
    --eval-dataset training/eval-dataset.jsonl \
    --predictions outputs/predictions-suddenly-7b.jsonl \
    --output results/eval-suddenly-7b.json

# 3. Avec BERTScore (optionnel, nécessite torch)
python pipeline/evaluate.py \
    --eval-dataset training/eval-dataset.jsonl \
    --predictions outputs/predictions-suddenly-7b.jsonl \
    --output results/eval-suddenly-7b.json \
    --bertscore
```

### Format du fichier de prédictions

Même structure que `eval-dataset.jsonl`, avec les contenus `assistant` remplacés par les sorties du modèle. Les champs `system` et `user` sont identiques à la référence.

```json
{
  "messages": [
    {"role": "system", "content": "Tu es un conteur..."},
    {"role": "user", "content": "Mon personnage entre dans la taverne..."},
    {"role": "assistant", "content": "RÉPONSE DU MODÈLE ICI"}
  ],
  "meta": {"genre": "medieval-fantastique", "turns": 1}
}
```

### Génération des prédictions avec vLLM

```python
from vllm import LLM, SamplingParams
import json

llm = LLM("./outputs/suddenly-7b")
params = SamplingParams(temperature=0.7, max_tokens=512, stop=["</s>"])

refs = [json.loads(l) for l in open("training/eval-dataset.jsonl")]

predictions = []
for ref in refs:
    # Construire le prompt depuis les messages (sans la dernière réponse assistant)
    msgs = ref["messages"]
    # Trouver la position de la dernière réponse assistant
    prompt_msgs = msgs[:-1] if msgs[-1]["role"] == "assistant" else msgs
    # Formater selon le template Mistral v1
    # (utiliser mistral_common ou le tokenizer du modèle)
    prompt = format_mistral_prompt(prompt_msgs)
    output = llm.generate([prompt], params)[0].outputs[0].text
    pred = dict(ref)
    pred["messages"] = prompt_msgs + [{"role": "assistant", "content": output}]
    predictions.append(pred)

with open("outputs/predictions-suddenly-7b.jsonl", "w") as f:
    for p in predictions:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
```

---

## Comparaison inter-runs

Le fichier JSON produit par `evaluate.py` peut être comparé entre plusieurs runs :

```python
import json
baseline = json.load(open("results/eval-mistral-7b-base.json"))
finetuned = json.load(open("results/eval-suddenly-7b.json"))

delta_chrf = finetuned["summary"]["avg_chrf"] - baseline["summary"]["avg_chrf"]
print(f"Δ chrF++ : {delta_chrf:+.4f}")
```

---

## Références

- [chrF paper — Popovic 2015](https://aclanthology.org/W15-3049/)
- [BERTScore](https://github.com/Tiiiger/bert_score)
- [sacrebleu](https://github.com/mjpost/sacrebleu)
- [Dataset d'évaluation](../training/eval-dataset.jsonl)
- [Script d'évaluation](../pipeline/evaluate.py)
