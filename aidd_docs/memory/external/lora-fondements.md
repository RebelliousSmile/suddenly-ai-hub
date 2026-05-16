# Fondements et choix techniques des LoRA — Suddenly AI Hub

> **Statut** : Document de référence technique
> **Mise à jour** : 2026-05-15

---

## 1. Qu'est-ce qu'un LoRA ?

### Définition

**LoRA** (Low-Rank Adaptation) est une technique de fine-tuning "économique" pour les grands modèles de langage. Au lieu de réentraîner les milliards de paramètres du modèle de base, on **gèle les poids d'origine** et on n'entraîne qu'un petit jeu de paramètres additionnels — typiquement **0,1 à 1 %** de la taille du modèle.

### Mathématiques

Plutôt que de modifier directement une matrice de poids $W$, on apprend une décomposition en matrices de rang faible :

$$\Delta W = A \times B$$

où :
- $A \in \mathbb{R}^{d \times r}$
- $B \in \mathbb{R}^{r \times k}$
- $r \ll \min(d, k)$ est le **rank** (facteur de réduction)

Les poids finaux s'obtiennent par **addition additive** au moment de l'inférence :

$$W_{final} = W_{base} + \Delta W = W_{base} + A \times B$$

### Concrètement

| Aspect | Fine-tuning complet | LoRA |
|---|---|---|
| Paramètres modifiés | Tous (~7 milliards pour 7B) | ~0,1–1 % (quelques millions) |
| Taille du modèle | Plusieurs Go | Quelques Mo |
| Coût VRAM | Très élevé (8×A100 min.) | Faible (1×A100 possible) |
| Temps d'entraînement | Jours | Heures |
| Stockage | Modèle complet | Adapter portable, versionnable |
| Composabilité | ❌ Impossible | ✅ Plusieurs adapters sur la même base |

### Propriétés exploitées par Suddenly

- **Adapter compact** — quelqes Mo au lieu de plusieurs Go, versionnable et swappable
- **Coût d'entraînement réduit** — accessible via Together.ai (~$0,02/h)
- **Composabilité** — plusieurs LoRA spécialisés peuvent être combinés sur le même modèle de base

> **Analogie dev** : C'est l'équivalent d'un *patch surcouche* plutôt qu'un *fork complet* du modèle. On spécialise sans dupliquer la base.

---

## 2. Modèle de base

### Choix : `Qwen/Qwen2.5-7B-Instruct`

| Critère | Qwen2.5-7B | Llama-3.1-8B | Mistral-7B |
|---|---|---|---|
| Français | ✅ Excellent | ⚠️ Moyen | ⚠️ Moyen |
| Dialogue RP | ✅ Optimisé (Instruct) | ✅ Bon | ✅ Bon |
| Coût Tokens | ~$2,70/M tokens | ~$3,00/M tokens | ~$3,00/M tokens |
| Fine-tunable Together | ✅ | ✅ | ✅ |

**Pourquoi Qwen2.5-7B-Instruct :**
- Bien supérieur à Llama/Mistral en français (critère décisif)
- Variant "Instruct" optimisé pour le dialogue et les interactions
- Coût très bas sur Together.ai
- Disponible en fine-tuning LoRA sur Together.ai
- Taille raisonnable (7B) pour un bon équilibre qualité/coût/VRAM

### Fallback : `Llama-3.1-70B-Instruct-Turbo`

Moindre capacité en LoRA (70B = beaucoup plus de paramètres à adapter), mais utilisé comme fallback si Qwen ne convient pas.

---

## 3. Architecture : deux axes de spécialisation

### Axe 1 — Univers (Genre)

LoRA spécialisé sur le **monde fictionnel** : vocabulaire, lore, références, système de magie, technologie, culture.

| Exemples | Détails |
|---|---|
| `fantasy-medievale` | Épées, magie, féodalité, royaumes |
| `cyberpunk` | Techno, mégacorporations, implant |
| `steampunk` | Vapeur, engrenages, victorien |
| `horreur-gothique` | Vampires, moustiques, atmosphère sombre |

**Taille rank recommandée** : 32 (`lora_r: 32`, `lora_alpha: 64`)
→ Le lore est riche et nécessite plus de capacités d'adaptation.

### Axe 2 — Situation (Thématique de scène)

LoRA spécialisé sur le **rythme narratif** et le **ton** d'un type de scène.

| Exemples | Détails |
|---|---|
| `combat` | Rythme rapide, action, violence, tension physique |
| `romance` | Émotion, tension relationnelle, sensibilité |
| `intrigue` | Manipulation, mystère, sous-entendus, politique |
| `exploration` | Découverte, émerveillement, observateur |
| `politique` | Négociation, diplomatie, enjeux sociaux |
| `quotidien` | Détente, interactions sociales légères |

**Taille rank recommandée** : 16 (`lora_r: 16`, `lora_alpha: 32`)
→ Le style de scène est moins profond que le lore universel.

---

## 4. Composition des LoRA : Stacking

### Principe mathématique

Lors de l'inférence, les adapters LoRA s'additionnent **linéairement** sur le modèle de base :

$$W_{final} = W_{base} + \alpha_1 \cdot (A_1 \times B_1) + \alpha_2 \cdot (A_2 \times B_2)$$

où $\alpha_1$ et $\alpha_2$ sont les **multipliers** (poids) de chaque adapter.

### Stacking sur Together AI

Together AI supporte nativement le passage de **plusieurs adapter IDs** avec des **multipliers distincts** dans une seule requête API :

```
adapter_1_id + adapter_2_id
multiplier_1    multiplier_2
```

**Avantage :** Pas besoin de pré-fusionner les adapters. Le système gère l'addition des deltas en temps réel.

### Multipliers : calibrer l'impact

Les multipliers permettent d'ajuster l'influence relative de chaque adapter :

| Configuration | Multiplier univers | Multiplier scène | Résultat |
|---|---|---|---|
| Univers dominant | 1.5 | 0.8 | Lore fort, scène subtile |
| Scène dominante | 0.8 | 1.5 | Ton de scène marqué |
| Équilibré | 1.0 | 1.0 | Balance univers + scène |
| Univers seul | 1.0 | 0.0 | Pas de spécialisation scène |
| Scène seule | 0.0 | 1.0 | Pas de spécialisation univers |

> **Attention** : Des multipliers trop élevés (> 2.0) peuvent dégrader la qualité générale du modèle (catastrophic forgetting de la base).

---

## 5. Stratégie de fusion : Pre-merge vs Stacking

Together AI supporte le stacking LoRA en temps réel. Mais pour le déploiement sur des instances Suddenly autonomes (sans connexion au hub Together), nous devons aussi supporter le **pre-merge offline**.

### Mécanisme de Pre-merge Offline

Processus :
1. Entraîner `lora-{univers}` et `lora-{situation}` séparément.
2. À la demande d'un couple donné (ex: `fantasy-medievale` + `combat`) :
   - Charger les deux adapters depuis le stockage.
   - Additionner les delta weights ($\Delta W_{total} = \Delta W_{univers} + \Delta W_{situation}$).
   - Générer un nouvel adapter unique pré-merge.
3. Charger le modèle fusionné dans vLLM pour l'inférence.

**Outils** : Scripts Axolotl (`merge_lora.py`) ou PEFT pour le merge.

### Comparaison : Stacking vs Pre-merge

| Critère | Stacking (Together API) | Pre-merge (vLLM local) |
|---|---|---|
| Latence | Très faible | Léger délai au premier usage |
| Coût | Appels API Together | Zéro (hors entraînement) |
| Flexibilité | Multipliers ajustables en temps réel | Fixed weights après merge |
| Qualité | Identique | Identique |
| Auto-hébergé | ❌ (dépend de Together) | ✅ (vLLM) |
| Stabilité | Dépend du fournisseur | Stable, versionnable |

### Déploiement mixte recommandé

- **Hub Suddenly** : utilise le stacking Together en temps réel (flexibilité maximale, multipliers dynamiques).
- **Instances autonomes** : utilise le pre-merge (vLLM local, zéro dépendance externe).
- **Fallback** : si le pre-merge d'un couple n'existe pas, fallback sur un seul des deux adapters (hiérarchie §6).

---

## 6. Hiérarchie de fallback

Si un adapter n'est pas disponible, le système utilise le niveau suivant :

1. **Pre-merged** (`lora-{univers}-{situation}`) : Meilleure qualité, combiné les deux axes.
2. **Situation seule** (`lora-{situation}`) : Style de réponse cohérent, sans lore universel.
3. **Univers seul** (`lora-{univers}`) : Thème respecté, sans ton de scène spécifique.
4. **LoRA générique** (`lora-generic`) : Style neutre RP, sans spécialisation.
5. **Modèle de base** : Fallback ultime — performances générales sans adaptation.

---

## 7. Seuil de déclenchement

### Critère de volume

- **500 sessions** par catégorie (univers ou situation) avant de lancer un entraînement LoRA.
- Ce seuil vise à garantir suffisamment de diversité de données pour un fine-tuning stable.

### Pré-merge automatique

Dès que les deux adapters individuels existent pour un couple donné, le pré-merge est automatique :

```
fantasy-medievale + combat → fantasy-medievale-combat (pré-merge)
```

---

## 8. Configuration d'entraînement

### Paramètres Axolotl par axe

**Axe Univers (rank 32) :**

```yaml
base_model: "Qwen/Qwen2.5-7B-Instruct"
lora_r: 32
lora_alpha: 64
lora_dropout: 0.05
epochs: 3
batch_size: 4
learning_rate: 2e-4
```

**Axe Situation (rank 16) :**

```yaml
base_model: "Qwen/Qwen2.5-7B-Instruct"
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
epochs: 3
batch_size: 4
learning_rate: 2e-4
```

### Différences clés

- `lora_r` et `lora_alpha` plus élevés pour l'axe univers (plus de paramètres à apprendre)
- Paramètres identiques pour le reste (dropout, epochs, batch, learning rate)

---

## 9. Coûts

| Élément | Coût estimatif |
|---|---|
| Entraînement LoRA (Together.ai) | ~$0,02/heure |
| Temps d'entraînement estimé | 2–6h par adapter |
| Coût par LoRA univers (rank 32) | ~$0,04–$0,12 |
| Coût par LoRA situation (rank 16) | ~$0,04–$0,06 |
| Stacking API call | ~$0,005–$0,02 par requête |
| Pre-merge + vLLM local | Zéro (hors entraînement initial) |

---

## 10. Limites et compromis

### Ce qu'un LoRA fait bien

- Spécialiser le **vocabulaire** et le **registres de langue** (lore universel)
- Ajuster le **rythme** et le **ton** (style de scène)
- Approcher les performances d'un fine-tuning complet sur des tâches ciblées
- Composabilité : combiner plusieurs spécialisations

### Ce qu'un LoRA fait moins bien

- Réécrire profondément le comportement du modèle (raisonnement complexe, créativité nouvelle)
- Gérer des contextes très longs sans遗忘 de la base
- Adapter des formats de sortie radicalement différents sans fine-tuning dédié
- **Pas de "sections" conditionnelles** : un LoRA est un patch uniforme, pas un arbre de décision

---

## 11. Glossaire

| Terme | Définition |
|---|---|
| **LoRA** | Low-Rank Adaptation — fine-tuning léger par matrices de rang faible |
| **Adapter** | Fichier contenant les deltas appris par le LoRA (quelques Mo) |
| **Rank (r)** | Dimension des matrices A et B — plus r = plus de capacité mais plus de paramètres |
| **Alpha (α)** | Facteur de scaling — contrôle l'intensité de l'adaptation |
| **Pre-merge** | Fusion offline de plusieurs adapters en un seul fichier |
| **Stacking** | Combinaison en temps réel de plusieurs adapters avec multipliers |
| **Multiplier/Weight** | Facteur d'ajustement de l'influence d'un adapter (0.0 à ~2.0) |
| **vLLM** | Moteur d'inférence optimisé pour les modèles LLM |
| **Together AI** | Plateforme cloud offrant fine-tuning et inférence LoRA |
| **Axolotl** | Framework d'entraînement LoRA (utilisé pour le fine-tuning) |
