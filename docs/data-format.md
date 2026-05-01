# Format des données d'entraînement

**Issue** : #9 | **Date** : 2026-05-01

---

## Format

Les exemples d'entraînement sont au format **JSONL** (JSON Lines) — un objet JSON par ligne.

```json
{"messages": [{"role": "system", "content": "..."}, {"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}
```

### Champs

| Champ | Type | Requis | Description |
|---|---|---|---|
| `messages` | array | ✅ | Liste ordonnée des messages de la conversation |
| `messages[].role` | string | ✅ | `system`, `user`, ou `assistant` |
| `messages[].content` | string | ✅ | Contenu textuel du message |

### Règles de structure

- **Alternance obligatoire** : les rôles `user` et `assistant` doivent s'alterner strictement. Deux messages consécutifs du même rôle font échouer le template Mistral.
- **System optionnel** : si présent, le message `system` doit être le premier élément de `messages`.
- **Minimum** : au moins une paire `user` / `assistant` (system optionnel).
- **Exemple minimal valide** :

```json
{"messages": [{"role": "user", "content": "Décris l'entrée dans la taverne."}, {"role": "assistant", "content": "La porte grince sur ses gonds rouillés..."}]}
```

---

## Contraintes

### Longueur de séquence

| Phase | Contexte max | `sequence_len` recommandé |
|---|---|---|
| Phase 1 — Together.ai | 32 768 tokens | ≤ 4 096 (économie) |
| Phase 2 — Axolotl QLoRA (GPU 24 Go) | — | 2 048 – 4 096 |
| Phase 2 — Axolotl QLoRA (A100 40 Go) | — | jusqu'à 8 192 |

- Les séquences dépassant `sequence_len` sont **tronquées** par Axolotl, pas rejetées.
- Les sessions RP en français font typiquement 1 500 à 4 000 tokens — `sequence_len: 4096` est le paramètre cible Phase 2.
- Longueur minimale : **200 tokens** (en dessous, l'exemple apporte peu de signal d'entraînement).

### Langue

- **Français uniquement** pour le corpus principal Phase 0–2.
- Anglais accepté uniquement pour les system prompts techniques (ex. : instructions de format).
- Support multilingue FR/EN envisagé en Phase 3 — non implémenté.

### Contenu

- Les noms de joueurs et personnages doivent être **anonymisés** avant contribution (pipeline d'anonymisation, issue #10).
- Les données sensibles (adresses, informations personnelles identifiables) sont rejetées à l'ingestion.
- Un exemple peut contenir plusieurs échanges (multi-turns), mais toute la session doit tenir dans `sequence_len`.

---

## Gestion du `system` prompt

Le chat template natif de Mistral v0.3 ne supporte pas le rôle `system` en position autonome dans la structure `[INST]`. Le pipeline d'ingestion doit **fusionner le system dans le premier message user** :

```
[INST] {system_content}\n\n{user_content_1} [/INST] {assistant_content_1}</s>
```

Cette fusion est effectuée automatiquement par `mistral_common` et par le chat template Axolotl `mistral_v1`. Le corpus source peut conserver le champ `system` — la transformation est appliquée au moment du preprocessing Axolotl.

---

## Validation

### Commande de validation Axolotl

```bash
axolotl preprocess training/configs/suddenly-7b.yml --debug
```

Cette commande tokenise le dataset, vérifie :
- L'alternance des rôles
- Les séquences hors limite (`sequence_len`)
- Le masquage correct des labels (seuls les tokens `assistant` sont entraînés, les tokens `system`/`user` ont `label = -100`)
- Les rôles invalides ou manquants

### Script de validation standalone

Un script de pré-validation Python est prévu dans `pipeline/` (issue #10) pour vérifier le format JSONL avant l'envoi au pipeline de fine-tuning, sans nécessiter une installation Axolotl complète.

---

## Exemple complet

Session RP 3 tours avec system prompt :

```json
{
  "messages": [
    {
      "role": "system",
      "content": "Tu es un conteur de roleplay médiéval-fantastique. Réponds en français, dans un registre narratif et immersif. Décris les scènes avec sensorialité et rythme."
    },
    {
      "role": "user",
      "content": "Mon personnage, Aldric, pousse la porte de la taverne et scrute la salle."
    },
    {
      "role": "assistant",
      "content": "L'air chaud et enfumé vous enveloppe dès le seuil. Une douzaine de bougies de suif trouent l'obscurité, projetant des ombres dansantes sur les poutres noircies. Dans un coin, deux mercenaires se disputent à voix basse ; derrière le comptoir, une femme aux bras épais essuie des chopes sans vous quitter des yeux."
    },
    {
      "role": "user",
      "content": "Je m'approche du comptoir et commande une bière en observant discrètement les mercenaires."
    },
    {
      "role": "assistant",
      "content": "— Une pièce, étranger, dit-elle en faisant glisser la chope sans vous regarder. Les mercenaires, eux, ont remarqué votre intérêt : le plus grand pose sa main sur son couteau, sans hâte, comme par habitude."
    }
  ]
}
```

---

## Références

- [Axolotl — Dataset formats : chat_template](https://docs.axolotl.ai/docs/dataset-formats/conversation.html)
- [Axolotl — exemple QLoRA Mistral](https://github.com/OpenAccess-AI-Collective/axolotl/blob/main/examples/mistral/qlora.yml)
- [Together.ai — Data Preparation](https://docs.together.ai/docs/fine-tuning-data-preparation)
- [Mistral v0.3 — chat template discussion](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3/discussions/37)
