# Cas d'usage des modèles Suddenly AI

> Source : [Issues GitHub Suddenly #72–#89](https://github.com/RebelliousSmile/suddenly/issues?q=is%3Aissue+label%3Aai)
>
> Tous les cas d'usage ci-dessous sont demandés pour les modèles fine-tunés du projet Suddenly.

---

## 1. Économie & infrastructure

### 1.1 Configuration IA par l'administrateur (#72)

- **Qui** : Administrateur d'instance Suddenly
- **Pourquoi** : Décider si son instance consomme et/ou contribue au hub Suddenly
- **Variables d'environnement** :
  - `SUDDENLY_AI_ENABLED` (bool, défaut `false`) — activation globale
  - `SUDDENLY_AI_CONTRIBUTE` (bool, défaut `false`) — soumission des données au fine-tuning
  - `SUDDENLY_AI_HUB_URL` (string, défaut `https://ai.suddenly.social`)
  - `SUDDENLY_AI_MONTHLY_MUSES` (int, défaut `0`) — allocation mensuelle par utilisateur
- **Impact modèle** : Aucun — cette issue gère le déploiement, pas le modèle.
- **Coût en Muses** : Non applicable

---

### 1.2 Affichage du solde de Muses et de la contribution utilisateur (#75)

- **Qui** : Joueur utilisateur
- **Pourquoi** : Savoir combien de Muses il lui reste, le coût de chaque feature, et sa contribution au modèle communautaire
- **Éléments demandés** :
  - Solde affiché dans la barre de navigation
  - Tooltip sur chaque bouton IA avec le coût
  - Historique des transactions dans le profil
  - Section "Ma contribution" : sessions contribuées, badge "Contributeur"
- **Grille tarifaire demandée** :

| Feature | Muses |
|---|---|
| Suggestion courte (dialogue/action) | 1 |
| Dialogue complet | 3 |
| Description de scène | 3 |
| Analyse cohérence scène | 5 |
| Suggestion pensée intérieure | 5 |
| Analyse cohérence session complète | 10 |
| Génération résumé de session | 10 |
| Analyse narrative profonde | 20 |

- **Impact modèle** : Aucun — purement UX/économie.
- **Coût en Muses** : Non applicable

---

### 1.3 Tableau de bord admin : gestion des Muses et contribution (#74)

- **Qui** : Administrateur d'instance
- **Pourquoi** : Suivre la consommation de Muses, le budget, et la contribution de son instance au modèle communautaire
- **Éléments demandés** :
  - Statut du hub (disponible / dégradé / indisponible)
  - Total Muses distribuées et consommées ce mois
  - Consommation par utilisateur (top 10)
  - Nombre de sessions contribuées au modèle
  - Pourcentage de sessions opt-in
  - Allocation mensuelle configurable
  - Reset mensuel automatique configurable
  - Alerte indisponibilité hub configurable
- **Impact modèle** : Aucun — purement admin/moniteurs.
- **Coût en Muses** : Non applicable

---

### 1.4 Distribution manuelle de Muses (#86)

- **Qui** : Administrateur d'instance
- **Pourquoi** : Récompenser la participation ou corriger une erreur de distribution
- **Éléments demandés** :
  - Distribution individuelle ou en masse (tous les utilisateurs actifs)
  - Création automatique de `MuseTransaction` avec motif
  - Notification utilisateur lors d'une distribution
- **Impact modèle** : Aucun.
- **Coût en Muses** : Non applicable

---

### 1.5 Exposition des metadata IA dans NodeInfo / ActivityPub (#73)

- **Qui** : Annuaire d'instances Suddenly, fédérateurs
- **Pourquoi** : Permettre aux joueurs de choisir une instance basée sur ses capacités IA
- **Champs NodeInfo** :
  - `ai_enabled` : bool
  - `ai_contribute` : bool
  - `ai_hub_url` : string
  - `ai_sessions_contributed` : int (sessions opt-in soumises au hub)
  - `ai_muses_distributed_month` : int
- **Impact modèle** : Aucun.
- **Coût en Muses** : Non applicable

---

### 1.6 Opt-in contribution des sessions au modèle communautaire (#87)

- **Qui** : Joueur utilisateur
- **Pourquoi** : Choisir si sa session contribue au fine-tuning du modèle communautaire
- **Critères** :
  - Case opt-in à la publication de session
  - Opt-in modifiable a posteriori
  - Soumission asynchrone (Celery) au hub (si instance activée)
  - Compte de sessions contribuées visible dans le profil
  - Récompense : +2 Muses pour la première contribution
- **Données transmises au hub** :
  - Reports de la session (anonymisés, sans noms)
  - Type de chaque report (dialogue, action, description, etc.)
  - Langue détectée
- **Impact modèle** : Ces données anonymisées alimentent l'entraînement des adapters LoRA. C'est la boucle de rétroaction entre l'usage et l'amélioration du modèle.
- **Coût en Muses** : 0 (l'utilisateur est *récompensé*, pas débité)

---

## 2. Suggestions de contenu (génération)

### 2.1 Suggestion de dialogue pour un personnage (#77)

- **Qui** : Joueur en train d'écrire un compte-rendu RP
- **Pourquoi** : Obtenir une réplique cohérente avec le personnage
- **Déclencheur** : Bouton "✨ Suggérer" sur les reports de type `dialogue`
- **Contexte envoyé au modèle** :
  - Fiche du personnage (nom, description, statut PJ/PNJ)
  - N reports précédents
  - Scène courante (reports de type `description` récents)
- **Sortie attendue** : Suggestion de dialogue inline, première personne
- **Actions utilisateur** : Accepter, modifier, ou ignorer
- **Contraintes** :
  - Affiché inline, jamais inséré automatiquement
  - Grisé si solde insuffisant
- **Coût** : 1 Muse (suggestion courte) ou 3 Muses (dialogue complet)
- **Adapter correspondant** : `suddenly-dialogue`

---

### 2.2 Suggestion d'action pour un personnage (#78)

- **Qui** : Joueur en train d'écrire
- **Pourquoi** : Dynamiser l'écriture, rester fidèle au caractère du personnage
- **Déclencheur** : Bouton "✨ Suggérer" sur les reports de type `action`
- **Contexte envoyé au modèle** :
  - Fiche du personnage (nom, description, traits de caractère)
  - Derniers reports de la scène
- **Sortie attendue** : Suggestion d'action cohérente avec le ton et le style établis
- **Actions utilisateur** : Accepter, modifier, ou ignorer (même UX que la suggestion de dialogue)
- **Coût** : 1 Muse (courte) ou 3 Muses (développée)
- **Adapter correspondant** : `suddenly-action`

---

### 2.3 Suggestion de description de scène (#79)

- **Qui** : Joueur en train d'écrire
- **Pourquoi** : Enrichir le compte-rendu avec des éléments sensoriels et atmosphériques
- **Déclencheur** : Bouton "✨ Décrire" sur les reports de type `description`
- **Contexte envoyé au modèle** :
  - Actions et dialogues récents (pour déduire l'ambiance)
- **Sortie attendue** : Description d'ambiance ou de lieu, style RP
- **Variantes** :
  - Liée à un personnage (POV)
  - Narrative (sans personnage)
- **Coût** : 3 Muses
- **Adapter correspondant** : `suddenly-description`

---

### 2.4 Suggestion de pensée intérieure (#80)

- **Qui** : Joueur en train d'écrire
- **Pourquoi** : Enrichir la narration avec la vie psychologique du personnage
- **Déclencheur** : Nouveau type de report optionnel : `pensée` (accessible depuis le menu d'insertion)
- **Contexte envoyé au modèle** :
  - Fiche personnage
  - Échanges récents de la scène
- **Sortie attendue** : Monologue intérieur à la première personne, point de vue du personnage
- **Coût** : 5 Muses
- **Adapter correspondant** : `suddenly-thought`

---

## 3. Analyse de cohérence

### 3.1 Analyse de cohérence RP sur une scène (#81)

- **Qui** : Joueur avant de publier
- **Pourquoi** : Détecter les incohérences de caractère ou de ton
- **Déclencheur** : Bouton "🔍 Analyser la scène" au niveau de la session
- **Sortie attendue** : Rapport d'analyse (panneau latéral non bloquant) :
  - Incohérences de personnage détectées
  - Ruptures de ton narratif
  - Suggestions de corrections
- **Coût** : 5 Muses
- **Adapter correspondant** : `suddenly-consistency-scene`

---

### 3.2 Analyse de cohérence RP sur toute la session (#82)

- **Qui** : Joueur à la fin d'une session
- **Pourquoi** : Vérifier que les arcs personnages sont cohérents de bout en bout
- **Déclencheur** : Bouton "🔍 Analyser la session" sur la page de session
- **Sortie attendue** : Rapport incluant :
  - Résumé de l'arc narratif de chaque personnage
  - Incohérences détectées avec référence au report concerné
  - Suggestions de liens claim/adopt/fork pertinents entre PNJ
- **Coût** : 10 Muses
- **Adapter correspondant** : `suddenly-consistency-session`

---

### 3.3 Résumé de session (#83)

- **Qui** : Joueur souhaitant publier ou archiver sa session
- **Pourquoi** : Faciliter la publication et la découvrabilité par d'autres joueurs
- **Déclencheur** : Bouton "✨ Générer le résumé" sur la page de session
- **Sortie attendue** :
  - Résumé narratif éditable avant validation
  - Ton : narratif, troisième personne, style compte-rendu JDR
  - Remplace ou complète le champ résumé existant
- **Longueur cible** : ~500 mots
- **Coût** : 10 Muses
- **Adapter correspondant** : `suddenly-summary`

---

### 3.4 Suggestions de liens fédérés (#84)

- **Qui** : Joueur après analyse de session ou manuellement
- **Pourquoi** : Tisser des connexions narratives entre fictions indépendantes (PNJ d'une session ↔ personnages publics de l'instance)
- **Déclencheur** : Après analyse de session ou bouton dédié
- **Sortie attendue** :
  - Comparaison PNJ de la session ↔ personnages publics de l'instance
  - Suggestions classées par niveau de pertinence (fort / moyen / faible)
  - Lien direct vers claim/adopt/fork depuis la suggestion
- **Coût** : 20 Muses
- **Adapter correspondant** : `suddenly-federation`

---

## 4. Infrastructure & intégration

### 4.1 LLMClient : abstraction hub Suddenly (#76)

- **Qui** : Développeur (couche applicative Suddenly)
- **Pourquoi** : Centraliser tous les appels IA, découpler le code de l'infrastructure du hub
- **Emplacement** : `suddenly/ai/client.py`
- **Interface** :
  ```python
  class LLMClient:
      def suggest(
          self,
          feature: str,        # "dialogue", "action", "description", etc.
          context: list[Report], # reports précédents
          character: Character,  # personnage ciblé
          session: Session,      # session en cours
      ) -> SuggestionResult:
          ...
  ```
- **Caractéristiques** :
  - API compatible OpenAI (`/v1/chat/completions`)
  - Authentification via signature ActivityPub (pas de clé API en clair)
  - Sélection automatique du modèle selon la feature (voir §4.5)
  - Gestion erreurs : timeout, rate limit, hub indisponible
  - Logging : feature, tokens input/output, durée
  - Désactivation propre si `SUDDENLY_AI_ENABLED=False`
  - Exécution asynchrone via Celery
  - Cache Redis (TTL 5 min)
- **Impact modèle** : Définit l'interface contractuelle que les adapters doivent servir. Chaque feature mappe à un appel `LLMClient.suggest(feature=..., ...)`.
- **Coût en Muses** : Non applicable (couche technique)

---

### 4.2 Sélection automatique du modèle selon la feature (#85)

- **Qui** : Développeur / hub Suddenly
- **Pourquoi** : Choisir le modèle optimal (qualité / coût) pour chaque feature
- **Table de correspondance demandée** :

| Feature | Clé config | Muses |
|---|---|---|
| Suggestion courte | `suggest_short` | 1 |
| Dialogue | `suggest_dialogue` | 3 |
| Action | `suggest_action` | 3 |
| Description | `suggest_desc` | 3 |
| Pensée intérieure | `suggest_thought` | 5 |
| Analyse cohérence scène | `analyze_scene` | 5 |
| Analyse cohérence session | `analyze_session` | 10 |
| Résumé de session | `generate_summary` | 10 |
| Suggestions liens fédérés | `suggest_links` | 20 |

- **Comportement** :
  - Table `feature → coût Muses` en config Django
  - Le hub sélectionne automatiquement le modèle optimal
  - Si une feature nécessite un contexte que le modèle ne supporte pas → feature désactivée côté UI (masquée silencieusement)
- **Impact modèle** : Définit le mapping feature → adapter. C'est la correspondance directe avec les 8 adapters LoRA du projet.
- **Coût en Muses** : 1 à 20 selon la feature

---

### 4.3 Mode dégradé : feature IA indisponible (#88)

- **Qui** : Joueur utilisateur
- **Pourquoi** : Comprendre pourquoi une feature IA est indisponible sans être bloqué
- **Comportements demandés** :
  - Si `SUDDENLY_AI_ENABLED=false` : aucun bouton IA visible
  - Si solde Muses insuffisant : bouton grisé + message "X Muses nécessaires"
  - Si hub indisponible (timeout / erreur 5xx) : message "Assistant indisponible, réessayez plus tard"
  - Aucune Muse débitée si la requête échoue
  - Alerte admin si hub indisponible > 30 minutes (configurable)
- **Impact modèle** : Aucun — gestion d'erreurs côté hub.
- **Coût en Muses** : 0 (pas de débit en cas d'erreur)

---

### 4.4 Export de prompt scène pour générateur vidéo IA (#89)

- **Qui** : Joueur souhaitant visualiser sa scène
- **Pourquoi** : Générer un prompt structuré pour les générateurs vidéo (Sora, Runway, Kling, etc.) sans quitter Suddenly
- **Déclencheur** : Bouton "🎬 Exporter en prompt vidéo" sur une scène ou des reports sélectionnés
- **Sortie attendue** : Prompt texte structuré affiché dans une modale éditable, prêt à copier-coller, contenant :
  - Description visuelle du lieu et de l'ambiance
  - Description physique des personnages présents
  - Action principale de la scène
  - Ton cinématographique (style, lumière, cadrage suggéré)
- **Contrainte** : Aucun envoi vers un service externe — Suddenly génère uniquement le texte du prompt
- **Coût** : 5 Muses
- **Adapter correspondant** : `suddenly-description` (étendu à la synthèse visuelle)

---

## 5. Récapitulatif des adapters requis

| # | Feature | Adapter LoRA | Muses | Type |
|---|---|---|---|---|
| 77 | Suggestion de dialogue | `suddenly-dialogue` | 1–3 | Génération |
| 78 | Suggestion d'action | `suddenly-action` | 1–3 | Génération |
| 79 | Suggestion de description | `suddenly-description` | 3 | Génération |
| 80 | Suggestion de pensée intérieure | `suddenly-thought` | 5 | Génération |
| 81 | Analyse cohérence scène | `suddenly-consistency-scene` | 5 | Analyse |
| 82 | Analyse cohérence session | `suddenly-consistency-session` | 10 | Analyse |
| 83 | Résumé de session | `suddenly-summary` | 10 | Génération |
| 84 | Suggestions de liens fédérés | `suddenly-federation` | 20 | Analyse |
| 89 | Export prompt vidéo | `suddenly-description` (étendu) | 5 | Génération |

---

## 6. Flux de données : de l'usage au modèle

La boucle complète se décompose ainsi :

```
Joueur utilise feature IA (dialogue, action, etc.)
    │
    ▼
LLMClient.suggest(feature=..., context=...)  ← #76, #85
    │
    ▼
Hub sélectionne le bon LoRA adapter  ← #85
    │
    ▼
Réponse générée affichée inline  ← #77, #78, #79, #80
    │
    ▼
Joueur accepte/modifie/ignore
    │
    ├─→ Débiter X Muses du wallet  ← #75
    │
    └─→ Si opt-in activé (#87) :
         Session anonymisée soumise au hub (Celery)
         → Alimente le dataset d'entraînement LoRA
         → Amélioration collective du modèle
         → +2 Muses au contributeur
```
