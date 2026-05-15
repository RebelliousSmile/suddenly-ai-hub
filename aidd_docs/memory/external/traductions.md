# Méthodologie de Traduction de Gros Volumes (EN -> FR)

**2026-05-15** — Dernière mise à jour — **Pour usage interne et Passe Cyberpunk (#64)**

---

## 1. Objectif
Traduire et adapter massivement des corpus de dialogues (Visual Novels, scénarios de RP, romans SF) de l'anglais vers un français naturel, en conservant le registre, les idiomes et le ton (doux, dramatique, etc.).

## 2. Pipeline Technique

### A. Architecture du script de traduction
1. **Extraction du texte** : Isoler les dialogues et le contexte (genre, situation) dans le corpus source.
2. **Préparation des prompts** : Construire un prompt système standard + un prompt utilisateur pour chaque entrée.
3. **Appels API parallélisés** : Envoyer les requêtes en masse via `asyncio` ou `ThreadPoolExecutor` pour saturer le débit de l'API.
4. **Nettoyage et parsing** : Extraire le JSON de la réponse (souvent encadré de ` ```json `), le valider et le réinsérer dans le corpus de sortie.

### B. Prompt Engineering
Le prompt doit être **concis** pour limiter les tokens (coût et latence) tout en imposant une **structure de sortie stricte**.

**Modèle de prompt recommandé :**
> *System* : Tu es un traducteur spécialisé en jeux de rôle textuels français. Traduire et adapter un dialogue de Visual Novel de l'anglais vers un français de jeu de rôle naturel. Utilise le tiret français (-) pour les dialogues courts. Conserve la structure JSON originale. Ne fournis aucun commentaire.
>
> *User* : Genre: [genre], Situation: [situation].
> [JOUEUR]: {dialogue_joueur}
> [PERSONNAGE]: {dialogue_bot}

## 3. Fournisseurs et Modèles Testés (Retour d'expérience)

### A. Together AI
**Statut actuel :** ⚠️ **Restreint / Réorganisé**
Depuis le redéploiement de leurs infrastructures, Together AI a retiré la plupart des gros modèles (Mistral Large, Llama 3 70B) de l'accès serverless (pricing à la demande).

**Modèles restants disponibles en mode serverless :**
| Modèle | Vitesse | Qualité FR | Statut |
|--------|---------|------------|--------|
| `Qwen/Qwen2.5-7B-Instruct-Turbo` | ⚡ Ultra (0.5s/req) | Faible (style technique) | Utilisable pour dé-doublonnage ou pré-traitement, pas pour traduction créative |
| `meta-llama/Llama-3.3-70B-Instruct-Turbo` | 🐌 Lent (~60s/req) | ✅ Excellente | Rentabilité faible pour le volume (>10h de traduction pour 386 entrées) |
| `mistralai/Mistral-Large-2411` | — | — | ❌ **Disponible** (Erreur 404 sur l'API Together) |

*Note :* La plupart des modèles requis nécessitent désormais la création d'un **endpoint dédié** (coûteux à l'heure, pas adapté au batch léger).

### B. Solutions alternatives pour les gros volumes
- **Ollama (Local)** : ✅ **Recommandé**. La solution zéro-coût la plus robuste. `ollama pull mistral` ou `llama3.3` (70B quantifié) pour une traduction de qualité locale et gratuite.
- **OpenRouter** : Permet d'acheter de petites quantités de modèles mid-tier (ex: Mistral 7B, Llama 3 70B) au token. Bon compromis si le hardware local manque.

## 4. Recommandations pour la Passe Cyberpunk (#64)

Pour la traduction des corpus EN -> FR (traductions CC BY-NC-SA de Doctorow, Watts, etc.), voici la stratégie recommandée :

1. **Pré-traitement** : Utiliser un modèle rapide (Qwen 2.5-7B sur Together ou Ollama) pour la traduction *brute* (sens littéral) ou pour le dé-doublonnage préliminaire.
2. **Traduction créative (Style RP)** : Utiliser **Ollama en local** avec `mistral` ou `llama3.3` pour l'adaptation des idiomes, du registre SF/technologique et du ton.
3. **Post-traitement** : Script de validation pour s'assurer que les noms de personnages sont cohérents et que le format JSONL n'est pas corrompu.

## 5. Scripts et Outils Disponibles
- `scripts/crawl_rpv/translate_rpv_to_french.py` : Pipeline batch classique (Threading).
- `scripts/crawl_rpv/translate_rpv_to_french_async.py` : Pipeline asynchrone (`asyncio`/`httpx`) — *Plus performant pour les appels API massifs (gère 8 requêtes en parallèle).*

## 6. Limitations connues & Astuces
- **Limites de taille** : Toujours tronquer le `content` des messages système (>300-500 chars) pour ne pas saturer le budget `max_tokens`.
- **Parsing JSON** : Les grands modèles rajoutent souvent du markdown (```json ... ```). Un script de nettoyage robuste (suppression des backticks) est obligatoire avant le `json.loads()`.
- **Rate-limiting** : Toujours inclure un délai (`BATCH_DELAY`) entre les batchs pour éviter le bannissement de l'IP.

---
## 🆕 À venir : Traduction du Cyberpunk (#64)
- Corpus source : Traductions CC BY-NC-SA (Doctorow, Watts, etc.)
- Volume estimé : ~2M tokens
- Défi majeur : Gestion de la terminologie SF/Technologique (nécessite un glossaire en amont).
