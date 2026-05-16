# Fine-Tuning via Together.ai — Plan révisé

> **But :** Valider la chaîne complète validate → upload → train → infer → eval sur Together.ai (sans GPU local).
> **Révision :** Challenge fix (0 deal breakers). Réduction de 5 scripts à 2 scripts.

**Architecture :** Un seul script CLI `scripts/train_together.py` avec subcommands `validate`, `train`, `infer`. Évaluation via `pipeline/evaluate.py`.
**Tech Stack :** Python 3.12, Together.ai API, modèle fine-tunable (à vérifier), sacrebleu, pytest

---

## Contexte actuel
- API Together.ai opérationnelle (clé dans `.env`, modèle Qwen2.5-7B-Instruct-Turbo valide en inférence)
- Dataset test : `data/test-dataset-rp.jsonl` (10 exemples RP, format Axolotl sharegpt)
- `pipeline/evaluate.py` : évaluation chrF++, BERTScore, répétition, langue → prêt
- `scripts/train_model.py` : Axolotl local — obsolète
- `data/*.html` : HTML brute jdRoll (pas exploitable)

---

## Task 1 : Script unique `scripts/train_together.py` avec 3 subcommands

**Fichier :** `scripts/train_together.py` (remplace les 3 scripts initiaux)

### Subcommand `validate`

**Objectif :** Valider le JSONL en entrée + split train/val.

**Étapes :**
1. Lire le JSONL, valider chaque ligne (JSON valide, champ `messages` présent, structure correcte)
2. **Vérifier le system prompt** : si aucun message `system` trouvé dans un exemple, injecter un prompt par défaut (`Tu es un conteur de roleplay. Réponds en français, dans un registre narratif et immersif.`) — flag `--system <prompt>` pour personnaliser
3. Séparer 80% train / 20% val (random seed=42 pour reproductibilité)
4. Afficher stats : nombre d'exemples, répartition train/val, taille moyenne
5. Écrire 2 fichiers : `data/train.jsonl` et `data/val.jsonl`
6. Flag `--dry-run` pour validation sans écriture

### Subcommand `train`

**Objectif :** Upload du JSONL vers Together.ai, lancer le training.

**Étapes :**
1. Vérifier que le modèle est fine-tunable → utiliser une liste hardcodée de modèles connus fine-tunables (l'endpoint `GET /v1/models` ne documente pas un champ `fine_tuning`). Liste : `Qwen/Qwen2.5-7B-Instruct`, `meta-llama/Llama-3.1-70B-Instruct-Turbo`, `Qwen/Qwen2.5-14B-Instruct`. Si le modèle demandé n'est pas dans cette liste, utiliser le premier de la liste comme fallback.
2. Si le modèle n'est pas fine-tunable → erreur claire avec liste des modèles supportés
3. Upload via `POST /v1/files` (multipart/form-data, `purpose: fine-tune`, `file` = `data/train.jsonl`)
4. Créer training via `POST /v1/fine_tuning/jobs` avec les hyperparams (model, lr, epochs, hyperparameters)
5. Poll status via `GET /v1/fine_tuning/jobs/{id}` toutes les 30s
6. Afficher progression : `CHECKING` → `RUNNING` → `SUCCEEDED` / `FAILED`
7. En cas succès → persister le `result_model_id` dans `.cache/model_id.txt` et afficher le message

**Timeout :** Poll toutes les 30s, timeout par défaut 3600s (1h). Flag `--timeout=3600` pour personnaliser. Si timeout → erreur claire avec le status actuel.

**Hyperparams par défaut :**
- `model` : `Qwen/Qwen2.5-7B-Instruct`
- `learning_rate_multiplier` : 2.0
- `n_epochs` : 3

**Modèle fallback :** Si le modèle par défaut n'est pas fine-tunable, essayer dans l'ordre : `meta-llama/Llama-3.1-70B-Instruct-Turbo`, `Qwen/Qwen2.5-14B-Instruct`. Si aucun n'est disponible, liste exhaustive des modèles fine-tunables récupérée via `GET /v1/models` + flag `--list-models` pour afficher. Le flag `--model-id <id>` est un override optionnel, le script lit `.cache/model_id.txt` par défaut.

**Note :** Together.ai ne supporte PAS `batch_size`. Les seuls hyperparams sont `learning_rate_multiplier` et `n_epochs`. `model_specific_params` est optionnel pour des params spécifiques au modèle.

**Note coûts :** Together.ai fine-tuning charge ~$2.70/M tokens d'input. 8 exemples × ~300 tokens ≈ ~$0.02. Très faible.

### Subcommand `infer`

**Objectif :** Faire inférence sur les exemples val avec le modèle fine-tuné.

**Étapes :**
1. Lire `.cache/model_id.txt` pour récupérer le model_id (produit par `train`). Si absent, afficher erreur claire + demander d'exécuter `train` d'abord.
2. Flag `--model-id <id>` en override optionnel (si pas de fichier cache)
3. Lire `data/val.jsonl`
4. Pour chaque exemple, appeler `POST /v1/chat/completions` avec le model_id
5. Stocker les réponses dans `data/val-predictions.jsonl`

---

## Task 2 : Tests + nettoyage du repo

**Objectif :** S'assurer que tout est testé et le repo propre.

**Étapes :**
1. `pytest` tests pour `train_together.py` :
   - `test_validate_format` : JSON valide/invalid
   - `test_split_data` : vérifie ratio 80/20 avec seed=42
   - `test_infer_format` : vérifie structure des prédictions
2. Vérifier que `.cache/` est dans `.gitignore` (contient model_id, pas de secret mais convention de ne pas tracker)
3. Supprimer `scripts/train_model.py` (Axolotl obsolète, `rm -f`)
4. Supprimer `scripts/convert_to_jsonl.py` (dépend jdRoll, `rm -f`)
5. Supprimer les scripts jdroll obsolètes (`scripts/scraper_jdroll.py`, `scripts/session*.py`, `rm -f`)
6. Relancer `pytest` — 100% pass

**Commit :** `git add scripts/train_together.py && git commit -m "scripts: unified Together.ai fine-tuning client with validate/train/infer subcommands"`

---

## Task 3 : Premier run de validation

**Objectif :** Lancer le pipeline sur le dataset test et valider.

**Étapes :**
1. `python scripts/train_together.py validate --input data/test-dataset-rp.jsonl`
   → `data/train.jsonl` (8 exemples), `data/val.jsonl` (2 exemples)
2. `python scripts/train_together.py train`
   → upload, training, récupération du `result_model_id`
3. `python scripts/train_together.py infer` (lit `.cache/model_id.txt` auto)
   → prédictions sur les 2 exemples val
4. `python pipeline/evaluate.py --eval-dataset data/val.jsonl --predictions data/val-predictions.jsonl --output results/eval-result.json`
   → métriques sur données non vues → significatif
5. Si tout passe → pipeline validé

---

## Notes
- Les modèles fine-tunés sur Together.ai sont **uniquement accessibles via API** (pas de download local)
- Évaluation sur données non vues (val set) → métriques significatives
- Coût total estimé : ~$0.02 (training) + ~$0.01 (inférence) = ~$0.03 pour validation complète
- Pas de GPU nécessaire — tout cloud
- Le script inclut une vérification automatique de disponibilité fine-tuning avant de lancer
