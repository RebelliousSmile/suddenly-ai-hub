# Stratégie d'adaptation LoRA

> **Source** : Issue #29 — [SPIKE] Valider l'architecture LoRA deux axes + pre-merge offline
> **Date** : 2026-05-15
> **Statut** : ✅ Décision actée

---

## 1. Architecture définitive

L'objectif est de spécialiser le modèle de base `suddenly-7b` (ou `13b`) en deux dimensions orthogonales pour offrir une expérience RP sur-mesure à chaque instance Suddenly.

### Modèle de base
- **Modèle** : `suddenly-7b` (ou `suddenly-13b`) — fine-tuné sur corpus RP général.
- **Format** : Peft LoRA adapter.

### Axes de spécialisation (LoRA)

#### Axe 1 — Univers (Genre)
LoRA spécifique au genre du RP.
- **Exemple** : `lora-medieval-fantastique`
- **Taille rank** : 32 (`lora_r: 32`, `lora_alpha: 64`)

#### Axe 2 — Situation (Thématique)
LoRA spécifique à la situation narrative (action, romance, enquête, etc.).
- **Exemple** : `lora-combat`
- **Taille rank** : 16 (`lora_r: 16`, `lora_alpha: 32`)

---

## 2. Mécanisme de fusion (Pre-merge Offline)

vLLM ne supporte pas nativement le stacking de deux LoRA en temps réel. Nous utiliserons un **pre-merge offline**.

**Processus :**
1. Entraîner `lora-{univers}` et `lora-{situation}` séparément.
2. À la demande d'un utilisateur pour un couple donné (ex: `medieval-fantastique` + `combat`) :
   - Charger les deux adapters depuis le stockage.
   - Additionner les delta weights (différences par rapport au modèle de base).
   - Générer un nouvel adapter unique pré-merge.
3. Charger le modèle fusionné dans vLLM pour l'inférence.

**Outils Axolotl** : Utilisation des scripts Axolotl (`merge_lora.py` ou PEFT) pour le merge.

---

## 3. Chaîne de fallback (fallback hierarchy)

Si un adapter n'est pas disponible (pas assez de données, modèle non entraîné), le système utilise le niveau suivant :

1. **Pre-merged** (`lora-{univers}-{situation}`) : Meilleure qualité, combiné les deux axes.
2. **Situation seule** (`lora-{situation}`) : Style de réponse cohérent.
3. **Univers seul** (`lora-{univers}`) : Thème respecté.
4. **LoRA générique** (`lora-generique`) : Style neutre RP.
5. **Modèle de base** : Fallback ultime.

---

## 4. Seuil de déclenchement

- **500 sessions** par catégorie (genre ou situation) avant de lancer un entraînement LoRA.
- Le pré-merge est automatique dès que les deux adapters individuels existent pour un couple donné.

---

## 5. Implémentation

- **Config Axolotl** : Voir `training/lora-situation.yml` et `training/lora-univers.yml`.
- **Déploiement vLLM** : Chargement/déchargement des adapters.
- **Instance Suddenly** : Passage des paramètres `genre` + `situation` dans chaque requête API.
