# Audit v2 — Cohérence issues vs memory
Date: 2026-05-01

## Résultat global
Confidence: 98%
Incohérences trouvées: 3
Corrections appliquées: 3

## Contexte
Cet audit fait suite à l'audit v1 (`2026_05_issues_memory_coherence.md`) qui avait corrigé 5 incohérences.
Les 5 corrections v1 ont été vérifiées et sont toutes présentes dans le state actuel des issues.
3 incohérences résiduelles ont été identifiées et corrigées dans ce second passage.

## Issues corrigées

### #11 — Noms de fichiers config Axolotl avec préfixe `base-` incorrect
- **Problème** : L'issue proposait `training/base-7b.yml` et `training/base-13b.yml` comme noms de configs. Le préfixe `base-` contredit la naming convention actée dans `architecture.md` qui nomme les modèles `suddenly-7b` et `suddenly-13b` (pas de préfixe `base-`).
- **Correction** : Renommé en `training/suddenly-7b.yml` et `training/suddenly-13b.yml`.

### #22 — Milestone incorrect : Phase 1 au lieu de Phase 2
- **Problème** : Le déploiement progressif canary (10% → 50% → 100%) était assigné au milestone "Phase 1 — Beta ouverte". `project_brief.md` (section Roadmap) place explicitement le canary deployment en Phase 2 — Stabilisation.
- **Correction** : Milestone déplacé vers "Phase 2 — Stabilisation".

### #14 — Nommage adapters dans le fallback : `lora-{genre}` au lieu de `lora-{univers}`
- **Problème** : La logique de fallback dans l'issue utilisait `lora-{genre}-{situation}` et `lora-{genre}` comme noms d'adapters internes. `architecture.md` acte le nommage `lora-{univers}-{situation}` et `lora-{univers}` pour les adapters. `genre` est le nom du paramètre API côté client — pas le nom interne de l'axe.
- **Correction** : Fallback mis à jour avec `lora-{univers}` comme nom d'adapter, avec une note explicite distinguant le paramètre API (`genre`) du nommage interne (`univers`).

## Issues conformes
#6, #7, #8, #9, #10, #11 (après correction), #12, #13, #14 (après correction), #15, #16, #17, #18, #20, #21, #22 (après correction), #23, #24, #25, #26, #27, #28, #29, #30, #31, #32, #33

## Résidus non corrigibles via issues

### project_brief.md — Ambiguïté terminologique `genre` vs `univers`
Le fichier utilise indistinctement les deux termes pour désigner l'axe univers LoRA :
- Ligne 60 : "genre : axe univers GROG" — correct
- Mais le terme `genre` seul apparaît plusieurs fois sans préciser qu'il désigne l'axe univers
Cette ambiguïté n'est pas un bug fonctionnel mais peut générer des confusions entre le paramètre API (`genre`) et l'axe interne (`univers`). À clarifier dans la memory ou dans un ADR dédié.

### project_brief.md — Roadmap Phase 1 mentionne "Together.ai API integration"
La roadmap Phase 1 inclut "Together.ai API integration" alors que `deployment.md` présente Together.ai comme un outil Phase 0/1 transitoire (avant migration Axolotl en Phase 2). Pas d'incohérence stricte, mais la roadmap ne précise pas que c'est transitoire. Non corrigeable via issues — à clarifier dans la memory.
