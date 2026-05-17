# Corpus RP public — Sourcing pour le bootstrap des tables

**Issue** : #12 | **Date initiale** : 2026-05-01 | **Rewrite** : 2026-05-17 (post-pivot)

---

## Contexte

Ce document inventorie les sources de corpus RP / narratif francophones publiquement disponibles, leurs licences, et les commandes de téléchargement.

Ces corpus servent au **bootstrap** du service Muses : extraction d'entités, templates, beats et fragments candidats pour amorcer les tables (cf. `philosophy.md` §3 « Continu, pas batch » et `architecture-tables-ml.md` § Provenance des tables → 1. Mining).

> **Note** : avant le pivot, ce document décrivait le sourcing pour le fine-tune `suddenly-7b` et `suddenly-13b`. Cette finalité est abandonnée. Les sources elles-mêmes restent pertinentes mais leur exploitation passe désormais par le pipeline de mining vers les tables, pas par Axolotl ou un fine-tune.

---

## Sources recommandées

### 1. OPUS / OpenSubtitles FR (CC BY 4.0)

**URL** : https://opus.nlpl.eu/OpenSubtitles/corpus/version/OpenSubtitles
**Licence** : Creative Commons Attribution 4.0
**Volume** : ~2,7 M paires de phrases en FR
**Format** : TMX ou TSV (source/cible alignés)
**Intérêt pour Muses** : dialogues FR naturels en variété de registres — bon vivier pour les rows de niveau `fragment` et pour extraire des entités lexicales conversationnelles. Tagging à inférer (univers / situation / rapport_initial / voix / emotion_dominante).
**Limite** : peu de narration RP au sens strict, beaucoup d'échanges courts à filtrer.

```bash
wget "https://object.pouta.csc.fi/OPUS-OpenSubtitles/v2018/mono/fr.txt.gz" \
    -O data/raw/opensubtitles-fr.txt.gz
gunzip data/raw/opensubtitles-fr.txt.gz
```

### 2. Project Gutenberg — Fiction française (domaine public)

**URL** : https://www.gutenberg.org/browse/languages/fr
**Licence** : domaine public (œuvres antérieures à la limite légale de protection)
**Format** : TXT brut
**Intérêt pour Muses** : prose narrative française de qualité, registre littéraire. Source riche pour les niveaux `template` (squelettes descriptifs), `fragment` (extraits courts de description ou de dialogue) et `entity` (lexique d'objets, lieux, gestes, émotions). À tagger sur les axes `univers` (historique, fantastique, contemporain selon l'œuvre), `voix` (lyrique, solennel typiquement).
**Exemples** :
- *Les Trois Mousquetaires* — Dumas (ID 13951)
- *Vingt mille lieues sous les mers* — Verne (ID 5097)
- *Le Comte de Monte-Cristo* — Dumas (ID 17989)

```bash
wget "https://www.gutenberg.org/files/13951/13951-0.txt" \
    -O data/raw/gutenberg-trois-mousquetaires.txt
```

### 3. Visual Novels Ren'Py (sources GitHub publiques)

Cf. `corpus/renpy-methodology.md` pour la méthodologie de collecte. Volume estimé ~2M tokens.
Intérêt particulier pour les rows de niveau `fragment` de dialogue et `template` de description courte, avec un tagging d'univers souvent explicite (cyberpunk, fantastique, romance contemporaine).

### 4. HuggingFace — `jpacifico/French-Alpaca-dataset-Instruct-110K`

**URL** : https://huggingface.co/datasets/jpacifico/French-Alpaca-dataset-Instruct-110K
**Licence** : Apache 2.0
**Volume** : 110 k exemples instruction/réponse en FR
**Intérêt pour Muses** : paires instruction/réponse FR. Utile principalement pour extraire des `fragment` cohérents en français standard. Faible spécificité RP — filtrer fortement.

### 5. Forums JDR FR (manuel, sous CGU)

**Sites historiquement identifiés** : [scenariotheque.org](https://scenariotheque.org), [lapartiedejdr.fr](https://lapartiedejdr.fr), [irtuel.fr](https://www.irtuel.fr)
**Licence** : variable — vérifier les CGU et obtenir un accord explicite avant collecte.
**Note** : la décision éthique prise lors de la session 2026-05-15 (cf. `issues-analysis.md`) exclut le scraping des forums RP privés. Cette source n'est utilisable qu'avec accord explicite des administrateurs.

---

## Filtrage recommandé avant mining

| Critère       | Règle                                                           |
| ------------- | --------------------------------------------------------------- |
| Langue        | Français uniquement (détection via `langdetect` ou similaire)   |
| Longueur      | Filtrer les fragments < 30 caractères (non exploitables)        |
| Contenu       | Exclure URLs résiduelles, balises HTML, ponctuation aberrante   |
| Déduplication | Supprimer les quasi-doublons (MinHash ou hash exact)            |
| Anonymisation | Pipeline `pipelines/anonymization/` sur tous les noms propres   |

---

## Du corpus brut aux rows de tables

Le mining se déroule en plusieurs étapes (à formaliser dans un futur `mining-pipeline.md`) :

1. **Ingestion** : conversion vers texte brut UTF-8, segmentation en passages courts (phrase, paragraphe court, échange).
2. **Anonymisation** : remplacement des noms propres par des placeholders typés (`{char.name}`, `{place.name}`).
3. **Classification axiale** : tagger chaque passage sur les cinq axes canoniques (cf. `axes-and-tags.md`) — classifieur léger entraîné sur des exemples canoniques, ou tagging manuel pour les sources peu volumineuses.
4. **Extraction par niveau** :
   - **Entités** : NER (gestes, émotions, lieux, objets, traits) + clustering pour identifier les variantes (`serre les poings` / `serrent les poings`).
   - **Templates** : généralisation de phrases via remplacement de tokens spécifiques par des slots typés.
   - **Beats** : segmentation thématique d'une scène + classification du beat (hésitation, révélation, rebondissement…).
   - **Fragments** : extraction directe de répliques ou de phrases courtes prêtes à insérer.
5. **Validation à l'ingestion** : cf. `data-format.md` § Validation à l'ingestion.
6. **Insertion** : avec `source: bootstrap` (ou `source: mined` selon la voie d'extraction) — cf. `data-format.md` § schéma commun.

Les pipelines Python existants `pipelines/anonymization/` et `pipelines/crawl_rpv/` ont été conçus pour le pipeline LoRA et nécessitent adaptation pour produire des rows au format `data-format.md` plutôt que des conversations JSONL Axolotl.

---

## Références

- [OPUS corpus](https://opus.nlpl.eu)
- [Project Gutenberg FR](https://www.gutenberg.org/browse/languages/fr)
- `corpus/renpy-methodology.md` — collecte des VN Ren'Py
- `data-format.md` — format des rows produites
- `axes-and-tags.md` — taxonomie de tagging
