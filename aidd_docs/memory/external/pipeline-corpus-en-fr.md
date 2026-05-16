# Pipeline corpus EN -> FR pour LoRA stylistiques

**2026-05-16** — Méthode de travail consolidée. Complète `traductions.md` (historique des tentatives) et `lora-corpus-dp.md` (taxonomie genres + chemins).

---

## 1. Vue d'ensemble

Chaine en 4 etapes pour produire un corpus FR exploitable pour entrainer des LoRA stylistiques sur `suddenly-7b` (Qwen2.5-7B-Instruct) :

```
[sources EN]  ->  [NLLB 1.3B local]  ->  [scoring/curation]  ->  [(optionnel) benchmark LLM]  ->  [LoRA training]
```

Chaque etape est un script independant dans `scripts/crawl_rpv/`, communiquant par CSV. Tout tourne en local sur RTX 2080 Super (8 GB VRAM). Budget : 0 EUR.

---

## 2. Scripts et roles

| Script | Role | Entree | Sortie |
|---|---|---|---|
| `translate_csv_nllb.py` | Traduction massive EN->FR via NLLB 1.3B + classification + protection placeholders | CSV plat (`content`, optionnel `genre`/`situation`/`role`) | `{corpus}_en-to-fr_nllb-1.3B.csv` (ajoute `content_fr` + `_skip_reason`) |
| `score_corpus.py` | Curation : note chaque ligne sur 3 axes (qualite, balance, non-dup) | CSV traduit | `{stem}_scored.csv` trie par score |
| `benchmark_translation.py` | Comparaison side-by-side NLLB vs LLM Ollama sur N echantillons | CSV traduit | `{base}_benchmark_nllb_vs_{model}.csv` |
| `extract_cyberpunk_samples.py` | Prepare un mini-corpus de reference (Watts/Doctorow/Stross) pour stress-test traduction | Fichiers bruts dans `~/lora-fr/bench/cyberpunk/sources/` | `bench_cyberpunk_en.csv` |

---

## 3. Convention de chemins (rappel)

Voir `lora-corpus-dp.md` pour le detail. Resume :

- **Hors-repo** (`~/lora-fr/`, jamais commit) — pipeline genres litteraires :
  - `~/lora-fr/sources/<genre>/` : oeuvres brutes (PDF, HTML, EPUB)
  - `~/lora-fr/raw/<genre>/` : texte extrait
  - `~/lora-fr/clean/<genre>/` : nettoye (headers, OCR, reflow)
  - `~/lora-fr/chunks/<genre>/` : decoupe en passages
- **In-repo** (`data/`, gitignore mais sous suddenly.ai.hub) :
  - `data/litterature/<genre>.jsonl` : corpus final pret pour LoRA
  - `data/<corpus>_en-to-fr_nllb-1.3B.csv` : sorties pipelines intermediaires
  - `data/bench/<theme>/sources/` + `data/bench/<theme>/extracted/` : benchmarks specifiques (cf. cyberpunk). Petit volume, CC, referencables par les scripts via chemin relatif au repo -> on garde in-repo malgre la regle generale "bruts hors-repo".

**Pourquoi cette separation** : sources brutes (parfois lourdes, parfois sous licence floue) restent hors-repo ; seul le produit fini versionnable atterrit dans `data/`. Voir memoire `project_suddenly_ai_hub.md` : `data/` est gitignore et **sans backup distant** (S3 Hetzner juge trop cher, repo GitHub public).

---

## 4. Convention de nommage

Tous les artefacts intermediaires suivent : `{corpus}_{src}-to-{tgt}_{model-id}.csv`

Exemples :
- `renpy-corpus_en-to-fr_nllb-1.3B.csv` (sortie NLLB)
- `renpy-corpus_en-to-fr_nllb-1.3B_scored.csv` (apres scoring)
- `renpy-corpus_en-to-fr_benchmark_nllb_vs_mistral-nemo.csv` (apres benchmark)

Memoire associee : `feedback_naming_translation_outputs.md`.

---

## 5. Etape 1 : Traduction NLLB

```bash
python scripts/crawl_rpv/translate_csv_nllb.py <input.csv>
```

Comportement cle :
- Modele : `facebook/nllb-200-1.3B` (HuggingFace pipelines, FP16 sur GPU)
- Classification automatique de chaque ligne (genre, situation, role) si colonnes absentes
- Masquage des placeholders Ren'Py `[player]`, `{name}`, etc. avant traduction, demasquage apres
- Skip intelligent : marque `_skip_reason` pour lignes deja FR, code Ren'Py pur, vide, trop court
- Decoupe les passages longs (> SPLIT_THRESHOLD) en chunks pour eviter troncature NLLB
- Ecriture finale UNIQUEMENT en fin de run (pas de checkpoint incremental)

Contraintes pratiques :
- Sature les 8 GB VRAM (~7.9 Go observes)
- Bloque la GPU pour toute autre tache GPU pendant la duree du run
- Quelques heures pour un corpus de quelques milliers de lignes

---

## 6. Etape 2 : Curation par scoring

```bash
python scripts/crawl_rpv/score_corpus.py <translated.csv> [-w "0.5,0.3,0.2"] [--jaccard 0.85]
```

Trois axes, ponderes (defaut 0.5 / 0.3 / 0.2) :

1. **Qualite intrinseque** : longueur, ratio FR/EN, ponctuation finale, integrite placeholders, mots EN residuels, caracteres parasites.
2. **Balance** : poids log-inverse par (genre, situation) pour eviter qu'un genre domine.
3. **Non-duplication** : MinHash + LSH banding (NUM_PERM=64, LSH_BANDS=16, LSH_ROWS=4, SHINGLE_K=5), Jaccard 0.85 par defaut, union-find pour clusters, on garde le meilleur representant.

Sortie : CSV trie par score decroissant avec `score`, `score_quality`, `weight_balance`, `is_duplicate`, `dup_cluster`, `flags`. Aucun dependance externe (MinHash implemente en stdlib via `hashlib.blake2b`).

Usage typique : prendre le top-N (ex. top 70 %) pour le set d'entrainement LoRA, le reste pour test/holdout ou rejet.

---

## 7. Etape 3 (optionnelle) : Benchmark vs LLM

```bash
python scripts/crawl_rpv/benchmark_translation.py <translated.csv> -m mistral-nemo:12b-instruct-2407-q4_K_M -n 50
```

Sur un echantillon avec seed, retraduit chaque ligne via Ollama (prompt litteraire FR, temperature 0.35, garde placeholders), et produit un CSV side-by-side `(content_en, content_nllb, content_ollama)` pour jugement humain.

Sert a :
- Decider si NLLB suffit ou si une 2e passe LLM apporte un gain reel sur ce corpus
- Calibrer le choix de modele Ollama avant d'engager un run massif
- Mesurer le cout en temps (s/ligne) du LLM vs NLLB

Pre-requis : `ollama serve` lance, modele pulle (`ollama pull mistral-nemo:12b-instruct-2407-q4_K_M` ~7 GB).

---

## 8. Benchmark cyberpunk de reference

Pour valider que la chaine tient sur du contenu litteraire dense (le cyberpunk etant l'un des registres les plus durs : neologismes, syntaxe hachee, jargon tech) :

1. Sources telechargees en CC dans `data/bench/cyberpunk/sources/` :
   - Peter Watts, *Blindsight* (PDF officiel CC -> blindsight.pdf -> extrait via pdftotext en blindsight.txt)
   - Cory Doctorow, *Down and Out in the Magic Kingdom* (TXT CC -> down-and-out.txt)
   - Charles Stross, *Accelerando* (HTML auteur -> accelerando.html, parse BeautifulSoup)
2. `extract_cyberpunk_samples.py` reflow + filtre par longueur (600-1800 chars) et echantillonne N=5 passages par auteur avec seed.
3. Sortie : `data/bench/cyberpunk/extracted/bench_cyberpunk_en.csv` (colonnes `author, work, role, content`).
4. Passer ce CSV dans `translate_csv_nllb.py` puis `benchmark_translation.py`.

**Why** : c'est notre stress-test "vraie litterature". Si NLLB plante sur Watts mais tient sur Doctorow, on sait calibrer le seuil de qualite en consequence.

---

## 9. Contraintes transverses

- **Pas d'emojis dans les scripts** (cf. `feedback_no_emojis_in_scripts.md`).
- **Windows natif, pas WSL**.
- **PowerShell par defaut** ; Bash dispo via le tool Bash pour POSIX si besoin.
- **`data/` gitignore et sans backup distant** : avant tout `rm` ou reinstall, copier ailleurs (autre disque, archive locale).
- **GPU bloquee pendant un run NLLB** : pas de benchmark Ollama en parallele.

---

## 10. Ordre canonique d'execution

Pour un nouveau corpus EN -> dataset LoRA :

```bash
# 1. Traduction (long, sature GPU)
python scripts/crawl_rpv/translate_csv_nllb.py data/<corpus>.csv

# 2. Scoring (rapide, CPU)
python scripts/crawl_rpv/score_corpus.py data/<corpus>_en-to-fr_nllb-1.3B.csv

# 3. (optionnel) Benchmark sur 50 lignes (lent, Ollama)
ollama pull mistral-nemo:12b-instruct-2407-q4_K_M
python scripts/crawl_rpv/benchmark_translation.py data/<corpus>_en-to-fr_nllb-1.3B.csv -n 50

# 4. Selection top-N du scored.csv -> data/litterature/<genre>.jsonl
#    (etape manuelle pour l'instant, a scripter plus tard)
```
