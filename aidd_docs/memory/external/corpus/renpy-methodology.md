# Méthodologie de Collecte Corpus Ren'Py

**2026-05-15** — Dernière mise à jour

---

## Source

Visual Novels Ren'Py publics disponibles sur GitHub via releases ou dépôts open-source.

**Volume** : ~2M tokens estimés, ~386 entrées JSONL, 70 fichiers .rpy analysés.

---

## Pipeline

### 1. Découverte (github_search.py)

Recherche GitHub API avec requêtes ciblées :
- `language:Ren'Py` — langages Ren'Py
- `language:renpy` — variante minuscule
- `extension:.rpy` — fichiers Ren'Py
- `created:>2015` — projets récents
- `stars:>10` — projets populaires

Filtrage automatique des repos sans fichiers `.rpy` et repos outils (moteurs, frameworks).

**Résultat** : ~90 repos trouvés, 6 repos avec dialogues extractibles.

### 2. Extraction (extract_dialogues.py)

Pour chaque repo :
1. **Tri prioritaire** des fichiers .rpy (script > dialogue > act > chapter > scene)
2. **Téléchargement** via `raw.githubusercontent.com` (fallback API blob si échec)
3. **Parsing** des dialogues avec regex Ren'Py : `character "dialogue"`
4. **Conversion** en format Axolotl JSONL avec messages system/user/assistant

**Filtres qualité** :
- Min. 2 dialogues par scène
- Min. 100 tokens par conversation
- Ignorer `gui.rpy`, `screens.rpy`, `options.rpy` (UI uniquement)

### 3. Tagage automatique (run_pipeline.py)

Détection de genre à partir du nom du repo :

| Pattern | Genre | Situation |
|---------|-------|-----------|
| katawa, shoujo | romance | scolaire |
| ddlc, dolly | horreur | psychologique |
| learn, code, rpg | instruction | apprentissage |
| bytesoflove | romance | scolaire |
| danse, macabre | horreur | surnaturel |
| visualnovel, kit | instruction | développement |

---

## Statistiques du corpus

| Métrique | Valeur |
|----------|--------|
| Entrées | 386 |
| Tokens approx | 6 581 866 |
| Fichiers traités | 70 |
| Genres trouvés | 4 |

### Répartition par genre

| Genre | Entrées | % |
|-------|---------|---|
| romance | 276 | 72% |
| horreur | 82 | 21% |
| instruction | 9 | 2% |
| inconnu | 19 | 5% |

### Répartition par situation

| Situation | Entrées | % |
|-----------|---------|---|
| scolaire | 276 | 72% |
| psychologique | 72 | 19% |
| surnaturel | 10 | 3% |
| apprentissage | 7 | 2% |
| développement | 2 | 1% |

---

## Limitations

1. **Langue** : 100% anglais — les repos publics Ren'Py sur GitHub sont majoritairement en anglais
2. **Couverture genre** : Romance et horreur surreprésentés (DDLC, Katawa Shoujo dominent)
3. **Duplication** : ~647 messages dupliqués (surtout messages système avec "Genre: inconnu")
4. **Entrées courtes** : ~5% des entrées ont < 4 messages

### Pour améliorer

- Ajouter des repos Ren'Py en français (recherche GitHub ciblée FR)
- Augmenter le nombre de repos (limité à 20 par défaut, passer à 100)
- Ajouter un filtre anti-duplication basé sur le contenu (similarity hash)
- Classifier Ollama pour affiner les genres détectés

---

## Fichiers

| Fichier | Rôle |
|---------|------|
| `scripts/crawl_rpv/github_search.py` | Découverte repos GitHub |
| `scripts/crawl_rpv/extract_dialogues.py` | Extraction + conversion JSONL |
| `scripts/crawl_rpv/run_pipeline.py` | Orchestration + tagage |
| `scripts/crawl_rpv/README.md` | Guide utilisateur |
| `data/renpy-repos.json` | Liste des repos découverts |
| `data/renpy-corpus-final.jsonl` | Corpus final |

---

## Reproductibilité

```bash
# Exécuter le pipeline complet
python scripts/crawl_rpv/run_pipeline.py \
    --output data/renpy-corpus.jsonl \
    --max-repos 20 \
    --min-tokens 100
```

Nécessite : Python 3.12+, `requests`, token GitHub (`GH_TOKEN` dans `.env`).
