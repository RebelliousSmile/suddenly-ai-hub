# Feuille de route — Corpus français domaine public pour LoRA d'axe univers

> **Statut** : Document de travail
> **Mise à jour** : 2026-05-16

## Contexte projet

Ce pipeline alimente l'**axe « univers »** du stacking 3 axes décrit dans
[`lora-strategy.md`](./lora-strategy.md) (univers × situation × voix). Il
produit un dataset stylistique par genre, utilisé pour entraîner un LoRA
`lora-<genre>` via [`training/lora-univers.yml`](../../../training/lora-univers.yml)
sur le modèle de base `suddenly-7b` (Qwen2.5-7B-Instruct fine-tuné RP général).

Budget zéro. Corpus français domaine public uniquement.
Cible : **100-130k mots utiles par genre après curation** (proxy hors-RP du
seuil « 500 sessions » qui s'applique aux corpus issus de sessions Suddenly).

## Taxonomie

Les genres entraînables doivent être ceux acceptés par
[`training/lora-univers.yml`](../../../training/lora-univers.yml) (taxonomie
GROG, voir [legrog.org/themes](https://legrog.org/themes)) :

| Genre GROG               | Source DP envisagée                                          |
|--------------------------|--------------------------------------------------------------|
| `medieval-fantastique`   | Schwob, France (*La Révolte des anges*), Gautier, Nodier     |
| `scifi` + `space-opera`  | Corpus commun : Verne, Rosny aîné, Renard, Le Rouge          |
| `contemporain-fantastique` | Maupassant (*Le Horla*), Lorrain, Villiers, Maurice Level  |
| `cyberpunk`              | Pas de socle DP — Le Rouge / Renard en proxy faible, reporter |
| `post-apocalyptique`     | Rosny aîné *La Mort de la Terre*, Le Rouge proches            |
| `historique-fantastique` | Gautier *Le Capitaine Fracasse*, Féval *Le Bossu*, Zévaco    |
| `super-heros`            | Pas de socle DP — reporter                                    |
| `oriental-manga`         | Pas de socle DP — reporter                                    |
| `contemporain`           | Polar/aventure DP — Leblanc, Leroux, Souvestre & Allain      |
| `generique`              | Fallback, agrégat multi-genres                                |

**Notes de mapping** :
- L'« horreur » du mémo initial est rangée en `contemporain-fantastique`.
- **`scifi` et `space-opera` partagent un corpus DP commun.** Les voyages
  interplanétaires de Verne (*De la Terre à la Lune*, *Hector Servadac*),
  la SF préhistorique/cosmique de Rosny aîné (*Les Navigateurs de l'infini*,
  *La Mort de la Terre*) et les pulps français début XXe (Le Rouge —
  *Le Prisonnier de la planète Mars*) mélangent les deux registres. On peut
  entraîner deux LoRA distincts sur le même JSONL, ou un seul `scifi` qui
  sert d'alias pour les deux avec une chaîne de fallback (cf.
  [`lora-strategy.md`](./lora-strategy.md) §3). Distinction fine à valider
  par évaluation, pas par découpage du corpus en amont.
- « Steampunk » n'existe pas dans la taxonomie GROG ; les œuvres de Robida
  / Verne tardif peuvent enrichir `scifi` ou `historique-fantastique`
  plutôt que créer un genre hors taxonomie.
- « Polar » et « aventure » ne sont pas des univers mais des **situations**
  — leurs corpus iraient plutôt nourrir des LoRA d'axe situation
  ([`training/lora-situation.yml`](../../../training/lora-situation.yml)),
  pas univers.

## Séparation data brutes / repo

Pour ne pas mélanger les sources initiales (epub, txt bruts, chunks
intermédiaires) avec ce qui doit être versionné, on cloisonne :

| Emplacement                                    | Contenu                                |
|------------------------------------------------|----------------------------------------|
| `~/lora-fr/sources/<genre>/`                   | epub téléchargés (DP)                  |
| `~/lora-fr/raw/<genre>/`                       | txt issus de la conversion             |
| `~/lora-fr/clean/<genre>/`                     | txt nettoyés                           |
| `~/lora-fr/chunks/<genre>.json`                | chunks segmentés + classifiés          |
| `suddenly.ai.hub/data/litterature/<genre>.jsonl` | **seul artefact** versionné/utilisé pour l'entraînement |
| `suddenly.ai.hub/scripts/litterature/*.py`     | pipeline (versionné)                   |

`~/lora-fr/` est le répertoire de travail hors-repo (historiquement VPS,
peut être local). Le repo `suddenly.ai.hub` ne reçoit que le **résultat
final** prêt à l'entraînement, plus les scripts.

## Format de sortie attendu pour entraînement

LoRA d'axe univers = adaptation **stylistique** (continued pretraining
domain-adapté), pas SFT instruction-réponse. Format JSONL minimal :

```jsonl
{"text": "Long passage de prose en français, 600-1500 mots..."}
{"text": "Autre passage..."}
```

À placer dans `suddenly.ai.hub/data/litterature/<genre>.jsonl`. Référencé
par `DATASET_PATH` dans la copie de `training/lora-univers.yml` substituée
pour chaque genre.

## Phase 0 — Environnement Windows

Le poste de travail tourne sous Windows (RTX 2080 SUPER, voir
[`project_memory.md`](./project_memory.md) si présent). Tous les scripts
ci-dessous se lancent depuis Git Bash (livré avec Git for Windows).

```powershell
# Outils — via Chocolatey ou installeurs officiels
choco install calibre pandoc jq -y

# Python : utiliser le venv du projet
.\venv\Scripts\Activate.ps1
pip install beautifulsoup4 lxml httpx
```

Arborescence à créer — **hors du repo** pour les sources et
intermédiaires, **dans le repo** pour le final :

```bash
# Depuis Git Bash
mkdir -p ~/lora-fr/{sources,raw,clean,chunks}/{contemporain-fantastique,scifi,medieval-fantastique,historique-fantastique,post-apocalyptique,contemporain}

# Dans le repo, uniquement le dossier d'aboutissement
mkdir -p /c/Users/fxgui/Documents/Projets/MyApps/suddenly.ai.hub/data/litterature
```

## Phase 1 — Validation pipeline sur un genre pilote

**Genre pilote : `contemporain-fantastique`** (corpus court, stylistiquement
marqué, faible risque de désaccord taxonomique).

### 1.1 Téléchargement des epub

Source principale : https://beq.ebooksgratuits.com

| Auteur | Œuvre |
|---|---|
| Maupassant | *Le Horla et autres contes fantastiques* |
| Jean Lorrain | *Histoires de masques* |
| Villiers de l'Isle-Adam | *Contes cruels* |
| Erckmann-Chatrian | *Contes fantastiques* |
| Maurice Level | *Les Portes de l'enfer* |

Déposer dans `~/lora-fr/sources/contemporain-fantastique/`.

### 1.2 Conversion epub -> txt

```bash
cd ~/lora-fr/sources/contemporain-fantastique
for f in *.epub; do
  ebook-convert "$f" "../../raw/contemporain-fantastique/${f%.epub}.txt" \
    --enable-heuristics --no-default-epub-cover 2>/dev/null
done
```

### 1.3 Nettoyage

Le code ci-dessous est à mettre dans `scripts/litterature/clean.py` (créer
le dossier en parallèle de `scripts/crawl_rpv/`). Aucun emoji dans les
scripts (cf. mémoire `feedback_no_emojis_in_scripts.md`).

```python
import re, sys
from pathlib import Path

def clean(text: str) -> str:
    text = re.sub(r'BeQ.*?\n', '', text)
    text = re.sub(r'La Bibliotheque electronique du Quebec.*?\n', '', text)
    text = re.sub(r'Collection.*?\n', '', text)
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'^\d+\.\s+.*$', '', text, flags=re.MULTILINE)
    text = text.replace('«\xa0', '« ').replace('\xa0»', ' »')
    text = re.sub(r'[—–]', '—', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    return text.strip()

def main():
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.mkdir(parents=True, exist_ok=True)
    for f in src.glob('*.txt'):
        (dst / f.name).write_text(clean(f.read_text(encoding='utf-8')), encoding='utf-8')
        print(f"OK {f.name}")

if __name__ == "__main__":
    main()
```

```bash
# Lancer depuis la racine du repo suddenly.ai.hub
python scripts/litterature/clean.py \
  ~/lora-fr/raw/contemporain-fantastique \
  ~/lora-fr/clean/contemporain-fantastique
```

### 1.4 Découpage en chunks

`scripts/litterature/chunk.py` — cible 1000 mots, plancher 600.

```python
import sys, json
from pathlib import Path

TARGET, MIN_W = 1000, 600

def chunk_text(text: str, source: str):
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    chunks, buf, count = [], [], 0
    for p in paragraphs:
        w = len(p.split())
        if count + w > TARGET and count >= MIN_W:
            chunks.append({'source': source, 'text': '\n\n'.join(buf)})
            buf, count = [p], w
        else:
            buf.append(p)
            count += w
    if buf and count >= MIN_W:
        chunks.append({'source': source, 'text': '\n\n'.join(buf)})
    return chunks

def main():
    src, out = Path(sys.argv[1]), Path(sys.argv[2])
    all_chunks = []
    for f in src.glob('*.txt'):
        all_chunks.extend(chunk_text(f.read_text(encoding='utf-8'), f.stem))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_chunks, ensure_ascii=False, indent=2), encoding='utf-8')
    print(f"OK {len(all_chunks)} chunks -> {out}")

if __name__ == "__main__":
    main()
```

```bash
python scripts/litterature/chunk.py \
  ~/lora-fr/clean/contemporain-fantastique \
  ~/lora-fr/chunks/contemporain-fantastique.json
```

### 1.5 Pré-classification via Ollama local

Objectif : filtrer pour garder uniquement `description` et `dialogue`, écarter
exposition lourde et notes.

Modèles candidats sur la 2080 SUPER 8 GB (voir [`benchmark-chats.md`](./benchmark-chats.md)
pour les retours d'expérience) :

- **`qwen3:8b`** (déjà pull) — choix par défaut pour démarrer.
- **`mistral-nemo:12b-instruct-2407-q4_K_M`** — meilleur français littéraire,
  VRAM tendue (~7-7.5 GB), recommandé pour la phase finale.

`scripts/litterature/classify.py` :

```python
import json, sys, httpx

MODEL = sys.argv[3] if len(sys.argv) > 3 else "qwen3:8b"
PROMPT = """Classe ce passage litteraire en UN seul mot parmi :
description, dialogue, action, exposition, monologue.

Passage :
{text}

Reponds uniquement par le mot."""

def main():
    src, dst = sys.argv[1], sys.argv[2]
    chunks = json.loads(open(src, encoding='utf-8').read())
    for c in chunks:
        r = httpx.post('http://localhost:11434/api/chat', json={
            'model': MODEL,
            'messages': [{'role': 'user', 'content': PROMPT.format(text=c['text'][:1500])}],
            'stream': False,
            'options': {'temperature': 0.1},
        }, timeout=120)
        c['type'] = r.json()['message']['content'].strip().lower().split()[0]
        print(f"{c['source'][:30]:30} -> {c['type']}")
    json.dump(chunks, open(dst, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)

if __name__ == "__main__":
    main()
```

```bash
ollama serve  # si pas deja lance
python scripts/litterature/classify.py \
  ~/lora-fr/chunks/contemporain-fantastique.json \
  ~/lora-fr/chunks/contemporain-fantastique_classified.json
```

### 1.6 Sélection finale + export Axolotl

```bash
# Filtrer description + dialogue (reste hors-repo)
jq '[.[] | select(.type == "description" or .type == "dialogue")]' \
  ~/lora-fr/chunks/contemporain-fantastique_classified.json \
  > ~/lora-fr/chunks/contemporain-fantastique_final.json

# Comptage mots utiles
jq -r '.[].text' ~/lora-fr/chunks/contemporain-fantastique_final.json | wc -w

# Export JSONL pour Axolotl, directement dans le repo (seul artefact versionné)
jq -c '.[] | {text: .text}' \
  ~/lora-fr/chunks/contemporain-fantastique_final.json \
  > data/litterature/contemporain-fantastique.jsonl
```

Relecture manuelle pour éjecter les faux positifs et l'exposition lourde
avant que `data/litterature/<genre>.jsonl` ne devienne `DATASET_PATH` dans
[`training/lora-univers.yml`](../../../training/lora-univers.yml).

## Phase 2 — Réplication sur les autres genres viables

Une fois le pilote validé, répliquer pour chaque genre du tableau de la
section taxonomie qui a une **source DP suffisante**.

### `scifi` (+ alias `space-opera`, même corpus)
- Verne — *Vingt mille lieues sous les mers*, *L'Île mystérieuse*,
  *De la Terre à la Lune*, *Hector Servadac*
- Rosny aîné — *Les Xipéhuz*, *La Mort de la Terre*,
  *Les Navigateurs de l'infini*
- Renard — *Le Docteur Lerne*, *Le Péril bleu*
- Le Rouge — *Le Prisonnier de la planète Mars*,
  *La Guerre des vampires* (suite)

### `medieval-fantastique`
- Schwob — *Le Roi au masque d'or*, *Vies imaginaires*
- France — *La Révolte des anges*, *Thaïs*
- Gautier — *Le Roman de la momie*
- Nodier — *Smarra, Trilby et autres contes*

### `historique-fantastique`
- Féval — *Le Bossu*
- Zévaco — *Les Pardaillan* (t.1)
- Gautier — *Le Capitaine Fracasse*
- Benoit — *L'Atlantide*
- Verne — *Michel Strogoff*

### `contemporain` (pour situations polar/aventure)
- Leblanc — *Arsène Lupin gentleman cambrioleur*, *813*
- Leroux — *Le Mystère de la chambre jaune*, *Le Parfum de la dame en noir*
- Souvestre & Allain — *Fantômas* (t.1)
- Gaboriau — *L'Affaire Lerouge*

### `post-apocalyptique`
- Rosny aîné — *La Mort de la Terre*
- Le Rouge — proches thématiquement, à vérifier

## Phase 3 — Genres reportés

Pas faisable sans génération synthétique payante ou licences :
- `cyberpunk` — socle DP français quasi-inexistant
- `super-heros`, `oriental-manga` — corpus DP francophone insuffisant

(`space-opera` n'est pas dans cette liste : couvert par le corpus `scifi`,
cf. taxonomie ci-dessus.)

À reporter quand budget API disponible (Together / Fireworks pour génération
synthétique — voir [`benchmark-fireworks-vs-together.md`](../benchmark-fireworks-vs-together.md)).

## Checklist par genre

- [ ] Sources DP téléchargées dans `~/lora-fr/sources/<genre>/`
- [ ] Conversion epub -> txt OK (`~/lora-fr/raw/<genre>/`)
- [ ] Nettoyage appliqué (`~/lora-fr/clean/<genre>/`)
- [ ] Chunks générés (1000 mots cible, 600 minimum)
- [ ] Classification effectuée (qwen3:8b ou mistral-nemo)
- [ ] Sélection manuelle terminée (description + dialogue conservés)
- [ ] Volume cible atteint (100-130k mots utiles)
- [ ] Export JSONL `suddenly.ai.hub/data/litterature/<genre>.jsonl` (seul artefact versionné)
- [ ] Substitution dans copie de `training/lora-univers.yml`
- [ ] Entraînement LoRA lancé (`axolotl train`)
