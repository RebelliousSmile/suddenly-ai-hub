# Méthodologie de Traduction de Gros Volumes (EN -> FR)

**2026-05-16** — Dernière mise à jour — **Guide de référence complet — Pour usage interne et Passe Cyberpunk (#64)**

---

## 1. Objectif
Traduire et adapter massivement des corpus de dialogues (Visual Novels, scénarios de RP, romans SF) de l'anglais vers un français naturel, en conservant le registre, les idiomes et le ton (doux, dramatique, etc.).

**Contrainte budget :** 0€ — exécution locale uniquement.

---

## 2. Méthodes Essayées — Retour d'Expérience Complet

### 🔴 Méthode 1 : Together AI + Llama 3.3-70B (3 passes)

**Date :** 2026-05-15
**Entrées :** 386 conversations Ren'Py
**Coût :** ~3€
**Durée :** ~60-90 minutes

#### Architecture
3 passes successives sur les mêmes 386 entrées :
1. **Pass 1 (Brute)** : Traduction littérale EN->FR
2. **Pass 2 (Style)** : Affinage stylistique (registres, idiomes, ton RP)
3. **Pass 3 (Nettoyage)** : Validation JSON, correction des champs vides/malformés

#### Résultats
| Pass | Succès | Erreurs | Taux |
|------|--------|---------|------|
| 1 | 302/386 | 84 HTTP 429 | 78% |
| 2 | 302/386 | 84 HTTP 429 | 78% |
| 3 | CRASH | JSONDecodeError sur champs vides | — |

#### Problèmes identifiés
- **Rate-limiting strict** : 8 workers provoquent des 429. Solution : réduction à 3 workers + `BATCH_DELAY=2.0s`.
- **Latence variable** : ~60s/req en serverless. 3 workers = ~20s effective/entrée.
- **Parsing JSON fragile** : Le modèle retourne du JSON vide ou entouré de texte, crashant `json.loads()` en Pass 3.
- **Sortie format nested invalide** : La structure `{messages: [], metadata: {}}` est **complètement altérée** par le modèle. Sortie : `{"SCÈNE": ...}` au lieu de JSON structuré. Contenus FR non alignés avec EN. 386 entrées **inutilisables pour LoRA**.

#### Décision
> ❌ **Abandonnée.** Le format JSON nested + 3 passes LLM = le modèle réorganise les clés et mélange les dialogues. Résultat : désalignement source/cible et hallucinations multilingues.

---

### 🔴 Méthode 2 : Ollama Local + Llama/Mistral (3 passes)

**Date :** 2026-05-15
**Entrées :** 386 conversations Ren'Py
**Coût :** 0€
**Durée :** ~Échec rapide

#### Problèmes identifiés
- **Instabilité en batch** : Les modèles conversationnels (Llama, Mistral via Ollama/Together) ne respectent pas les contraintes de nombre de lignes par requête, causant des désalignements critiques.
- **Résultat : désalignement source/cible** — chaque entrée EN ne correspond plus à sa traduction FR correspondante.

#### Décision
> ❌ **Abandonnée.** Les LLM (générateurs de texte) hallucinent et réorganisent le contenu. Impossible de garantir l'alignement 1:1 nécessaire pour un corpus d'entraînement.

---

### 🟡 Méthode 3 : NLLB Local (`facebook/nllb-200-distilled-600M`)

**Date :** 2026-05-16 (en cours)
**Entrées :** 10 665 lignes (CSV plat)
**Coût :** 0€
**Statut :** ⏳ En cours d'exécution sur CPU

#### Pourquoi NLLB ?
- **Modèle de traduction pure** : Pas de génération de texte libre. Sortie prévisible et 100% alignée.
- **Pas de clés API** : 100% offline, coût 0€.
- **Spécialisé EN->FR** : Entraîné spécifiquement pour la traduction, pas pour la conversation.

#### Format CSV plat (décision critique)
- Ancien format JSON nested (1 entrée = plusieurs messages imbriqués) → abandonné car impossible à préserver fidèlement par un LLM.
- Nouveau format : **1 tour de dialogue = 1 ligne CSV** (`genre`, `situation`, `role`, `content`).
- Garantit un alignement 1:1 entre source et cible.

#### Configuration
```python
src_lang="eng_Latn"  # Anglais
tgt_lang="fra_Latn"  # Français
batch_size=16
```

#### Environnement
- Python 3.12, PyTorch 2.12.0+cu130, transformers 4.x
- **CPU uniquement** — RTX 2080 Super non détectée par WSL2 (problème de pilote pont)
- Estimation : 1-2 heures restantes sur CPU

#### Décision
> 🟡 **En cours.** La méthode la plus fiable pour l'alignement, mais lente sur CPU.

---

## 3. Décision Tree — Quelle Méthode Utiliser ?

```
Besoin de traduction EN->FR massives ?
├── Volume < 500 entrées ?
│   └── ✅ Together AI + Llama 3.3-70B (3 passes)
│       - Acceptable si rate-limiting géré (3 workers, delay 2s)
│       - Coût ~3€ pour 386 entrées
│       - Format JSON nested → à éviter (modèle réorganise)
│
├── Volume > 500 entrées + GPU local ?
│   └── ✅ NLLB (facebook/nllb-200-distilled-600M)
│       - 100% offline, 0€, prévisible
│       - Format CSV plat (1 tour = 1 ligne)
│       - Besoin d'un bon GPU pour vitesse acceptable
│
├── Volume > 500 + Pas de GPU ?
│   └── ⚠️ NLLB sur CPU
│       - Fonctionnel mais lent (1-2h pour 10k lignes)
│       - À lancer en background
│
└── Besoin de style RP/adaptation créative ?
    └── ✅ Ollama local (Llama/Mistral) sur ENTRÉES FLATTES
        - Jamais sur format nested JSON
        - Pas en batch >1 (désalignement garanti)
        - Acceptable pour validation manuelle de 50-100 entrées
```

---

## 4. Format de Sortie Validé

### CSV Plat (Format Unique Valide)
```csv
genre,situation,role,content
romance,introduction,JOUEUR,Hey, how are you today?
romance,introduction,PERSONNAGE,I'm fine, thank you for asking.
horror,découverte,JOUEUR,What was that sound?
horror,découverte,PERSONNAGE,I don't want to tell you...
```

**Règle :** 1 tour de dialogue = 1 ligne. Colonnes obligatoires : `genre`, `situation`, `role`, `content`.

### Pourquoi ce format ?
- **Alignement 1:1 garanti** — aucune ambiguïté source/cible
- **Modèle ne réorganise pas** — NLLB traduit ligne par ligne
- **Compatible LoRA** — format Axolotl directement exportable

---

## 5. État Actuel

### Fichiers de données
| Fichier | Lignes | Statut |
|---------|--------|--------|
| `data/renpy-corpus-flat.csv` | 10 665 | ✅ Source complète (généré depuis 386 conversations) |
| `data/renpy-corpus-nllb-fr.csv` | En cours | ⏳ NLLB en cours d'exécution sur CPU |
| `data/renpy-corpus-final.jsonl` | 386 | ❌ 386 entrées inutilisables (format nested corrompu) |

### Environnement
| Composant | Statut |
|-----------|--------|
| Python 3.12 | ✅ |
| PyTorch 2.12.0+cu130 | ✅ installé |
| GPU CUDA (RTX 2080 Super) | ❌ WSL2 — pont pilote non configuré |
| NLLB 600M | ✅ disponible |

### Tickets associés
| # | Titre | Statut |
|---|-------|--------|
| #58 | Scraper Visual Novels Ren'Py depuis GitHub | ✅ Terminé (386 conversations → 10 665 lignes) |
| #64 | Phase 3 : LoRA Cyberpunk via traduction | ⏸️ En attente (ce fichier documente la méthode) |

---

## 6. Limitations connues & Astuces

- **Limites de taille** : Toujours tronquer le `content` des messages système (>300-500 chars) pour ne pas saturer le budget `max_tokens`.
- **Parsing JSON** : Les grands modèles rajoutent souvent du markdown (```json ... ```). Un script de nettoyage robuste (suppression des backticks) est obligatoire avant le `json.loads()`.
- **Rate-limiting** : Toujours inclure un délai (`BATCH_DELAY`) entre les batchs pour éviter le bannissement de l'IP.

---

## 7. Checklist Validation Qualité (pour NLLB)

Une fois la traduction NLLB terminée :
1. **Audit aléatoire** : Vérifier 50-100 entrées manuellement (ton RP, idiomes, alignement)
2. **Vérification colonnes** : Confirmer que `content` FR correspond à `content` EN (pas de mélange de lignes)
3. **Détection doublons** : Lancer l'audit de vocabulaire existant sur le corpus FR
4. **Export JSONL** : Regrouper les lignes CSV en format Axolotl structuré pour l'entraînement

---

## 8. Pour la Passe Cyberpunk (#64)

**Corpus source :** Traductions CC BY-NC-SA (Doctorow, Watts, etc.)
**Volume estimé :** ~2M tokens
**Méthode retenue :** NLLB sur CSV plat (cette documentation)
**Défi majeur :** Terminologie SF/Technologique — nécessite un glossaire en amont ou une passe de post-traitement spécifique.
