# Corpus RP public — Sourcing et fine-tuning initial

**Issue** : #12 | **Date** : 2026-05-01

---

## Contexte

Ce document décrit les sources de corpus RP public disponibles en français, leur licence, et les commandes pour préparer les données et lancer le fine-tuning `suddenly-7b` et `suddenly-13b`.

Il s'agit du fine-tuning initial (Phase 0) — avant que des données réelles de sessions Suddenly soient disponibles. L'objectif est d'inculquer au modèle le registre narratif et le rythme du RP FR, pas d'apprendre des comportements Suddenly spécifiques.

---

## Sources recommandées

### 1. OPUS / OpenSubtitles FR (CC BY 4.0)

**URL** : https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles  
**Licence** : Creative Commons Attribution 4.0  
**Volume** : ~2,7 M paires de phrases en FR  
**Format** : TMX ou TSV (source/cible alignés)  
**Intérêt** : dialogues FR naturels (films, séries), registre varié  
**Limite** : peu de narration RP ; nécessite filtrage des échanges trop courts

```bash
# Télécharger le sous-corpus FR (fichier TSV)
wget "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/fr.txt.gz" \
    -O data/raw/opensubtitles-fr.txt.gz
gunzip data/raw/opensubtitles-fr.txt.gz
```

### 2. Project Gutenberg — Fiction française (domaine public)

**URL** : https://www.gutenberg.org/browse/languages/fr  
**Licence** : domaine public (œuvres antérieures à 1928)  
**Format** : TXT brut  
**Intérêt** : prose narrative française de qualité, registre littéraire  
**Exemples** :
- *Les Trois Mousquetaires* — Dumas (ID 13951)
- *Vingt mille lieues sous les mers* — Verne (ID 5097)
- *Le Comte de Monte-Cristo* — Dumas (ID 17989)

```bash
# Exemple : télécharger un roman
wget "https://www.gutenberg.org/files/13951/13951-0.txt" \
    -O data/raw/gutenberg-trois-mousquetaires.txt

# Convertir en sessions narratives
python pipeline/format_corpus.py \
    --input data/raw/gutenberg-trois-mousquetaires.txt \
    --format narrative \
    --output data/corpus-gutenberg.jsonl
```

### 3. HuggingFace — `jpacifico/French-Alpaca-dataset-Instruct-110K`

**URL** : https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K  
**Licence** : Apache 2.0  
**Volume** : 110 k exemples instruction/réponse en FR  
**Format** : Parquet / JSONL  
**Intérêt** : paires instruction/réponse FR, couvre le registre narratif  
**Limite** : pas spécifique RP ; à filtrer sur les instructions narratives

```bash
pip install datasets
python - <<'EOF'
from datasets import load_dataset
ds = load_dataset("jpacifico/French-Alpaca-dataset-Instruct-110K", split="train")
ds.to_json("data/raw/french-alpaca.jsonl", force_ascii=False)
EOF

# Normaliser vers le format d'entraînement
python pipeline/format_corpus.py \
    --input data/raw/french-alpaca.jsonl \
    --format jsonl \
    --output data/corpus-alpaca.jsonl
```

### 4. Forums JDR FR (scraping manuel — vérifier CGU)

**Sites** : [scenariotheque.org](https://scenariotheque.org), [lapartiedejdr.fr](https://lapartiedejdr.fr), [irtuel.fr](https://www.irtuel.fr)  
**Licence** : variable selon le site — vérifier les CGU avant usage  
**Format** : HTML → texte après extraction  
**Intérêt** : contenu RP FR authentique, personnages et univers variés  
**Prérequis** : contacter les administrateurs pour accord d'utilisation

---

## Filtrage recommandé

Avant fine-tuning, appliquer les filtres suivants :

| Critère | Règle |
|---|---|
| Langue | Français uniquement (détecter avec langdetect) |
| Longueur | ≥ 200 tokens par session (~150 mots) |
| Alternance | Rôles user/assistant strictement alternés |
| Contenu | Exclure URLs, balises HTML résiduelles, textes tronqués |
| Déduplications | Supprimer les sessions quasi-identiques (MinHash ou hash exact) |

---

## Pipeline de préparation

```bash
# 1. Convertir les sources brutes
python pipeline/format_corpus.py \
    --input data/raw/ \
    --format dialogue \
    --glob "*.txt" \
    --output data/corpus-dialogue.jsonl

# 2. Anonymiser les noms propres (issue #10)
python -c "
import json, sys
from pipeline.anonymize import anonymize_session

with open('data/corpus-dialogue.jsonl') as fin, \
     open('data/corpus-rp-general.jsonl', 'w') as fout:
    for line in fin:
        obj = json.loads(line)
        obj['messages'] = anonymize_session(obj['messages'])
        fout.write(json.dumps(obj, ensure_ascii=False) + '\n')
"

# 3. Valider avec Axolotl (nécessite Axolotl installé)
axolotl preprocess training/suddenly-7b.yml --debug
```

---

## Lancement du fine-tuning

### Prérequis RunPod

- Template : `axolotl-runpod` (Axolotl + vLLM pré-installés)
- GPU : A100-40G (40 GB VRAM)
- Volume : Network Volume monté sur `/workspace`
- Secrets : `HF_TOKEN`, optionnellement `WANDB_API_KEY` et `S3_BUCKET`

### Étapes

```bash
# 1. Cloner le repo sur le pod
git clone https://github.com/RebelliousSmile/suddenly-ai-hub /workspace/suddenly-ai-hub
cd /workspace/suddenly-ai-hub

# 2. Copier le corpus préparé
# (upload via RunPod File Browser ou aws s3 cp)
cp data/corpus-rp-general.jsonl training/data/corpus-rp-general.jsonl

# 3. Fine-tuning suddenly-7b
./training/scripts/runpod_train.sh training/suddenly-7b.yml

# 4. Fine-tuning suddenly-13b
./training/scripts/runpod_train.sh training/suddenly-13b.yml
```

### Validation manuelle des sorties

Après entraînement, générer quelques exemples de prompts RP et évaluer manuellement :

```bash
# Avec vLLM (si pré-installé dans l'image RunPod)
python -c "
from vllm import LLM, SamplingParams

llm = LLM('./outputs/suddenly-7b')
params = SamplingParams(temperature=0.8, max_tokens=300)

prompts = [
    '[INST] Tu entres dans une taverne médiévale animée. Décris la scène. [/INST]',
    '[INST] Un elfe mystérieux te remet une carte au trésor. Que se passe-t-il ? [/INST]',
]
outputs = llm.generate(prompts, params)
for o in outputs:
    print(o.outputs[0].text)
    print('---')
"
```

**Critères d'évaluation** :
- Le modèle répond en français
- Le registre est narratif et immersif (pas conversationnel générique)
- Les personnages sont cohérents sur plusieurs tours
- Pas de répétitions excessives ni de troncatures brutales

---

## Références

- [OPUS corpus](https://opus.nlpl.eu)
- [Project Gutenberg FR](https://www.gutenberg.org/browse/languages/fr)
- [Axolotl — runpod_train.sh](../training/scripts/runpod_train.sh)
- [Format d'entraînement](data-format.md)
