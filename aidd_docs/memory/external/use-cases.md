# Cas d'usage du service Muses

> Source : [Issues GitHub Suddenly #72–#89](https://github.com/RebelliousSmile/suddenly/issues?q=is%3Aissue+label%3Aai)
>
> Ce document spécifie les cas d'usage **fonctionnels** demandés pour Muses, le service d'assistance créative mutualisé du Fediverse Suddenly. Sa mécanique interne (tables + ML, pipeline 4-étages) est définie dans `aidd_docs/memory/architecture-tables-ml.md` et docs adjacents. Le système monétaire (anciennement « Muses » comme monnaie, désormais « unité d'usage ») est en cours de refonte — les coûts par feature ne sont volontairement plus annotés ici et seront redéfinis dans un futur document de tarification.

---

## 1. Économie et gouvernance d'instance

### 1.1 Configuration Muses par l'administrateur d'instance (#72)

- **Qui** : Administrateur d'une instance Suddenly
- **Pourquoi** : Activer la connexion de son instance au service Muses partagé et choisir si elle contribue au pool de rows commun
- **Variables d'environnement** (anciens `SUDDENLY_AI_*` renommés `SUDDENLY_MUSES_*`) :
  - `SUDDENLY_MUSES_ENABLED` (bool, défaut `false`) — activation globale
  - `SUDDENLY_MUSES_URL` (string) — endpoint du service Muses (un seul, mutualisé)
  - `SUDDENLY_MUSES_CONTRIBUTE` (bool, défaut `false`) — opt-in instance pour soumission de rows candidates
  - `SUDDENLY_MUSES_ALLOCATION_MONTHLY` (int, défaut `0`) — allocation mensuelle d'unités d'usage par utilisateur (détails dans la refonte tarifaire)
- **Impact sur Muses** : Aucun — configuration côté instance.

---

### 1.2 Affichage du solde et de la contribution utilisateur (#75)

- **Qui** : Joueur utilisateur
- **Pourquoi** : Connaître son solde d'unités d'usage, le coût de chaque feature, sa contribution au pool de rows
- **Éléments demandés** :
  - Solde affiché dans la barre de navigation
  - Tooltip avec coût sur chaque bouton Muses (à figer dans la refonte tarifaire)
  - Historique des transactions dans le profil
  - Section « Ma contribution » : nombre de rows contribuées par niveau (entités, templates, beats, fragments), badge Contributeur si seuil atteint
- **Grille tarifaire** : la grille initiale (1 à 20 Muses) est obsolète — l'inférence sur tables est quasi-gratuite (cf. `philosophy.md` §7). Une nouvelle grille couvrira plutôt l'orchestration et la curation, à définir dans le futur document de tarification.
- **Impact sur Muses** : Aucun — purement UX instance.

---

### 1.3 Tableau de bord admin — gestion des unités et contribution (#74)

- **Qui** : Administrateur d'instance
- **Pourquoi** : Suivre la consommation, le budget, et la contribution de son instance au pool de rows
- **Éléments demandés** :
  - Statut du service Muses (disponible / dégradé / indisponible)
  - Total d'unités distribuées et consommées ce mois
  - Consommation par utilisateur (top 10)
  - Nombre de rows contribuées par l'instance (par niveau et par axe)
  - Pourcentage de sessions opt-in
  - Réputation d'instance courante (cf. `learning-and-trust.md` §5, visible admin)
  - Allocation mensuelle configurable
  - Reset mensuel automatique configurable
  - Alerte indisponibilité du service Muses configurable
- **Impact sur Muses** : Aucun — purement admin/monitoring.

---

### 1.4 Distribution manuelle d'unités (#86)

- **Qui** : Administrateur d'instance
- **Pourquoi** : Récompenser la participation, corriger une erreur de distribution
- **Éléments demandés** :
  - Distribution individuelle ou en masse
  - Création automatique de transaction avec motif
  - Notification utilisateur lors d'une distribution
- **Impact sur Muses** : Aucun.

---

### 1.5 Exposition des metadata Muses dans NodeInfo / ActivityPub (#73)

- **Qui** : Annuaires d'instances Suddenly, fédérateurs
- **Pourquoi** : Permettre aux joueurs de choisir une instance selon ses capacités Muses
- **Champs NodeInfo** (anciens `ai_*` renommés `muses_*`) :
  - `muses_enabled` : bool
  - `muses_contribute` : bool
  - `muses_url` : string
  - `muses_rows_contributed` : int (rows soumises au service)
  - `muses_units_distributed_month` : int
- **Impact sur Muses** : Aucun.

---

### 1.6 Opt-in contribution des rows par session (#87)

- **Qui** : Joueur utilisateur
- **Pourquoi** : Permettre que les rows candidates dérivées de sa session (éditions sur les suggestions, contributions explicites) alimentent le pool partagé du service Muses
- **Critères** :
  - Case opt-in à la publication de session, modifiable a posteriori
  - Soumission asynchrone des rows candidates (mécanisme `accept_edited` cf. `style-coaching.md` §3 ; la row produite porte `source: derived_from_edit` au sens de `data-format.md` § Schéma commun à toutes les rows)
  - Compte de rows contribuées visible dans le profil
  - Récompense : bonus d'unités d'usage pour la première contribution acceptée (montant dans la refonte tarifaire)
- **Données transmises au service Muses** :
  - Rows candidates anonymisées (cf. `data-format.md` § validation à l'ingestion)
  - Tags axiaux dérivés du contexte de session
  - Signature HTTP ActivityPub de la soumission
- **Impact sur Muses** : alimente les tables (croissance row par row), nourrit le trust du contributeur (cf. `learning-and-trust.md` §4) et la réputation de l'instance source (cf. §5 du même doc). C'est la boucle de rétroaction principale entre usage et amélioration.

---

## 2. Suggestions de contenu (génération)

### 2.1 Suggestion de dialogue (#77)

- **Qui** : Joueur en train d'écrire un report RP
- **Pourquoi** : Obtenir une réplique cohérente avec le personnage
- **Déclencheur** : Bouton « Suggérer » sur les reports de type `dialogue`
- **Contexte envoyé au service Muses** :
  - Fiche du personnage (nom, description, statut PJ/PNJ)
  - N reports précédents
  - Reports de type `description` récents (ambiance courante)
  - Tags axiaux dérivés du contexte (cf. `axes-and-tags.md`)
- **Sortie attendue** : Suggestion de dialogue affichée inline, première personne, jamais insérée automatiquement
- **Actions utilisateur** : Accepter, accepter avec édition, rejeter « pas ici », rejeter « bonne idée pas maintenant », ignorer (cf. `style-coaching.md` §3)
- **Niveaux de tables dominants** : fragments et entités
- **Étages du pipeline actifs** : 1 / 2 / 3 / 4 (best-of-N standard)
- **Coût** : voir refonte tarifaire

---

### 2.2 Suggestion d'action (#78)

- **Qui** : Joueur en train d'écrire
- **Pourquoi** : Dynamiser l'écriture, rester fidèle au caractère du personnage
- **Déclencheur** : Bouton « Suggérer » sur les reports de type `action`
- **Contexte envoyé** : fiche personnage avec traits, derniers reports de la scène, tags axiaux
- **Sortie attendue** : Action cohérente avec ton et style établis, même UX que la suggestion de dialogue
- **Niveaux dominants** : beats + templates + entités (le recombinateur est central pour assembler le geste / l'émotion / la suite)
- **Étages actifs** : tous
- **Coût** : voir refonte tarifaire

---

### 2.3 Description de scène (#79)

- **Qui** : Joueur en train d'écrire
- **Pourquoi** : Enrichir le compte-rendu avec des éléments sensoriels et atmosphériques
- **Déclencheur** : Bouton « Décrire » sur les reports de type `description`
- **Contexte envoyé** : Actions et dialogues récents pour déduire l'ambiance, tags axiaux
- **Sortie attendue** : Description d'ambiance ou de lieu, style RP
- **Variantes** : liée à un personnage (POV) ou narrative (sans personnage)
- **Niveaux dominants** : templates (squelettes descriptifs) + entités (lieux, ambiances, objets, sensations)
- **Étages actifs** : tous
- **Coût** : voir refonte tarifaire

---

### 2.4 Pensée intérieure (#80)

- **Qui** : Joueur en train d'écrire
- **Pourquoi** : Enrichir la narration avec la vie psychologique du personnage
- **Déclencheur** : Nouveau type de report optionnel `pensée`, depuis le menu d'insertion
- **Contexte envoyé** : Fiche personnage, échanges récents, tags axiaux (notamment `emotion_dominante`)
- **Sortie attendue** : Monologue intérieur à la première personne, POV du personnage
- **Niveaux dominants** : beats (hésitation, doute, révélation intérieure) + templates + entités émotionnelles
- **Étages actifs** : tous
- **Coût** : voir refonte tarifaire

---

## 3. Analyse de cohérence et synthèse

> Ces features utilisent le **pipeline d'analyse inversé** (cf. `architecture-tables-ml.md` § Pipeline d'analyse — projection inversée) : le contenu utilisateur est projeté sur des tables de patterns, au lieu d'être tiré depuis des tables de contenu.

### 3.1 Analyse de cohérence sur une scène (#81)

- **Qui** : Joueur avant de publier
- **Pourquoi** : Détecter les incohérences de caractère ou de ton
- **Déclencheur** : Bouton « Analyser la scène » sur la session
- **Sortie attendue** : Rapport en panneau latéral non-bloquant :
  - Incohérences de personnage détectées
  - Ruptures de ton narratif
  - Suggestions de corrections
- **Tables de patterns** : types d'incohérence (caractère / ton), beats canoniques attendus par `(situation, rapport_initial, voix)`
- **Mécanisme** : classification multi-label par fragment de scène + détection de transitions abruptes
- **Coût** : voir refonte tarifaire

---

### 3.2 Analyse de cohérence sur toute la session (#82)

- **Qui** : Joueur en fin de session
- **Pourquoi** : Vérifier la cohérence des arcs personnages bout en bout
- **Déclencheur** : Bouton « Analyser la session » sur la page de session
- **Sortie attendue** : Rapport incluant :
  - Résumé de l'arc narratif de chaque personnage
  - Incohérences détectées avec référence au report concerné
  - Suggestions de liens claim/adopt/fork pertinents entre PNJ (renvoi vers #84)
- **Tables de patterns** : arcs narratifs canoniques par `univers / situation`, archétypes de personnages
- **Mécanisme** : matching séquentiel de beats + scoring d'arc
- **Coût** : voir refonte tarifaire

---

### 3.3 Résumé de session (#83)

- **Qui** : Joueur souhaitant publier ou archiver
- **Pourquoi** : Faciliter publication et découvrabilité
- **Déclencheur** : Bouton « Générer le résumé » sur la page de session
- **Sortie attendue** :
  - Résumé narratif éditable avant validation
  - Ton : narratif, troisième personne, style compte-rendu JDR
  - Longueur cible : ~500 mots
  - Remplace ou complète le résumé existant
- **Tables exploitées** : tables de patterns (beats) pour extraction côté analyse + tables de templates de résumé taguées `(univers, situation)` côté génération. Cas hybride entre génération et analyse.
- **Mécanisme** : projection de la session sur des templates de résumé adaptés au contexte, remplissage des slots par les beats et entités extraits
- **Coût** : voir refonte tarifaire

---

### 3.4 Suggestions de liens fédérés (#84)

- **Qui** : Joueur après analyse de session ou manuellement
- **Pourquoi** : Tisser des connexions entre PNJ de la session et personnages publics de l'instance
- **Déclencheur** : Après analyse de session, ou bouton dédié
- **Sortie attendue** :
  - Comparaison PNJ de session ↔ personnages publics de l'instance source
  - Suggestions classées par niveau de pertinence (fort / moyen / faible)
  - Lien direct vers claim/adopt/fork
- **Tables exploitées** : embeddings des personnages publics de l'instance (catalogue maintenu par chaque instance)
- **Mécanisme** : similarité d'embeddings entre fiches PNJ et fiches publiques, seuils par niveau de pertinence
- **Coût** : voir refonte tarifaire

---

## 4. Infrastructure et intégration

### 4.1 MusesClient — abstraction côté instance Suddenly (#76)

- **Qui** : Développeur (couche applicative côté instance Suddenly)
- **Pourquoi** : Centraliser tous les appels au service Muses partagé, découpler le code de l'instance de l'infrastructure Muses
- **Renommage** : ancienne `LLMClient` → `MusesClient` (cohérent avec `philosophy.md` §8 « pas un LLM »)
- **Emplacement suggéré** : `suddenly/muses/client.py` côté Suddenly (hors de ce repo)
- **Interface conceptuelle** :
  ```python
  class MusesClient:
      def suggest(
          self,
          feature: str,          # "dialogue", "action", "description", "thought", "video_prompt"
          context: SessionContext, # reports, fiche perso, tags axiaux
      ) -> SuggestionResult:     # candidats top-N + traçabilité (rows tirées, scores)
          ...

      def analyze(
          self,
          feature: str,          # "consistency_scene", "consistency_session", "summary", "federated_links"
          content: SessionContent, # scène ou session complète
          tags: AxialTags,
      ) -> AnalysisReport:        # patterns matchés, scores, suggestions
          ...
  ```
- **Caractéristiques** :
  - API REST propriétaire (pas OpenAI-compatible — cf. `philosophy.md` §7 et §8)
  - Auth par signature HTTP ActivityPub
  - Routage interne service-side vers les étages du pipeline selon la feature (cf. §4.2)
  - Gestion erreurs : timeout, rate limit, service indisponible
  - Logging : feature, taille du contexte, durée d'appel, ID des rows servies (pour traçabilité)
  - Désactivation propre si `SUDDENLY_MUSES_ENABLED=false`
  - Exécution asynchrone (Celery côté Suddenly recommandé)
  - Cache local (TTL court, 1-5 min) sur les requêtes identiques
- **Impact sur Muses** : Définit le contrat API que le service Muses doit servir. Spec détaillée du protocole à figer dans `infrastructure.md` à venir.

---

### 4.2 Routage feature → étages du pipeline côté service Muses (#85)

- **Qui** : Service Muses (interne)
- **Pourquoi** : Router chaque feature vers les étages et niveaux de tables pertinents
- **Plus de « modèle par feature »** — il n'y a pas plusieurs modèles dans Muses. Chaque feature mappe à un sous-ensemble d'étages du pipeline 4-étages (génération) ou à la projection inversée (analyse).
- **Mapping** : cf. `architecture-tables-ml.md` § Mapping feature → niveaux exploités et § Pipeline d'analyse — projection inversée. Source unique, ne pas dupliquer ici.
- **Comportement de fallback** : si une cellule de la carte de couverture est sous-peuplée pour les tags du contexte, l'étage 1 (sélecteur) relâche les axes dans l'ordre `emotion_dominante → voix → rapport_initial → situation → univers` jusqu'à atteindre un seuil de peuplement (cf. `architecture-tables-ml.md` § Carte de couverture).
- **Impact sur Muses** : aucun nouveau, c'est de la doc d'architecture interne.

---

### 4.3 Mode dégradé — service Muses indisponible (#88)

- **Qui** : Joueur utilisateur
- **Pourquoi** : Comprendre pourquoi une feature Muses est indisponible sans être bloqué
- **Comportements demandés** :
  - Si `SUDDENLY_MUSES_ENABLED=false` : aucun bouton Muses visible
  - Si solde insuffisant : bouton grisé + message « N unités nécessaires »
  - Si service Muses indisponible (timeout / erreur 5xx) : message « Assistant indisponible, réessayez plus tard »
  - Aucune unité débitée si la requête échoue
  - Alerte admin si service indisponible > 30 minutes (configurable)
- **Impact sur Muses** : aucun — implémentation côté instance.

---

### 4.4 Export prompt vidéo (#89)

- **Qui** : Joueur souhaitant visualiser sa scène
- **Pourquoi** : Générer un prompt structuré pour les générateurs vidéo (Sora, Runway, Kling, etc.) sans quitter Suddenly
- **Déclencheur** : Bouton « Exporter en prompt vidéo » sur une scène ou un sous-ensemble de reports
- **Sortie attendue** : prompt texte structuré affiché dans une modale éditable, prêt à copier-coller, contenant :
  - Description visuelle du lieu et de l'ambiance
  - Description physique des personnages présents
  - Action principale de la scène
  - Ton cinématographique (style, lumière, cadrage suggéré)
- **Contrainte** : aucun envoi vers un service externe — Muses produit uniquement le texte du prompt, le joueur l'utilise comme il veut
- **Niveaux dominants** : templates visuels + entités (lieux, ambiances, objets, traits physiques)
- **Étages actifs** : 1 / 2 / 3 (canevas fixe d'assemblage, **pas de best-of-N**, donc étage 4 court-circuité — cf. `architecture-tables-ml.md` § Étage 4)
- **Coût** : voir refonte tarifaire

---

## 5. Récapitulatif features → niveaux / étages

| #   | Feature                          | Type        | Niveaux dominants                              | Étages actifs            |
| --- | -------------------------------- | ----------- | ---------------------------------------------- | ------------------------ |
| 77  | Suggestion dialogue              | Génération  | fragments + entités                            | 1 / 2 / 3 / 4            |
| 78  | Suggestion action                | Génération  | beats + templates + entités                    | 1 / 2 / 3 / 4            |
| 79  | Description de scène             | Génération  | templates + entités (lieux/ambiances)          | 1 / 2 / 3 / 4            |
| 80  | Pensée intérieure                | Génération  | beats + templates + entités émotionnelles      | 1 / 2 / 3 / 4            |
| 81  | Cohérence scène                  | Analyse     | tables de patterns (incohérence + beats canoniques) | projection inversée |
| 82  | Cohérence session                | Analyse     | arcs narratifs canoniques + archétypes         | projection inversée      |
| 83  | Résumé de session                | Hybride     | patterns (extraction) + templates de résumé    | projection + génération  |
| 84  | Suggestions liens fédérés        | Analyse     | embeddings personnages publics                 | similarité + seuils      |
| 89  | Export prompt vidéo              | Génération  | templates visuels + entités                    | 1 / 2 / 3 (sans 4)       |

---

## 6. Flux de données — de l'usage à la croissance des tables

```
Joueur déclenche une feature Muses dans son instance Suddenly
   │
   ▼
MusesClient.suggest(...) ou .analyze(...) ← #76
   │ (signature ActivityPub)
   ▼
Service Muses — routage feature → étages ← #85
   │
   ├─ génération : étages 1 (sélecteur) → 2 (pondérateur) → 3 (recombinateur) → 4 (filtreur)
   │
   └─ analyse    : embedder → matcher → agrégateur → rapport
   │
   ▼
Réponse renvoyée à l'instance ← #77-80, #81-84, #89
   │
   ▼
Joueur réagit via les 5 signaux UI (cf. style-coaching.md §3)
   │
   ├─→ Débit éventuel de l'unité d'usage ← #75 (refonte tarifaire)
   │
   └─→ Si opt-in instance ET opt-in session ← #87 :
         row candidate dérivée de l'édition / contribution explicite
            │ (anonymisation, tags axiaux, signature)
            ▼
         Soumission asynchrone au service Muses
            │
            ▼
         Validation à l'ingestion ← data-format.md § validation
            │
            ▼
         Ajout à la table appropriée + update trust contributeur
            │ ← learning-and-trust.md §4
            ▼
         Online learning des étages 1 / 2 / 4 ← learning-and-trust.md §3
```

---

## Pas dans le périmètre de ce document

- Mécanique interne du service Muses (pipeline 4-étages, tables, ML) — `architecture-tables-ml.md`.
- Format physique des rows — `data-format.md`.
- Taxonomie des valeurs sur les cinq axes — `axes-and-tags.md`.
- Online learning, trust, garde-fous décentralisés — `learning-and-trust.md`.
- Modes confort / challenge, profil de style, méta-suggestions, signaux UI — `style-coaching.md`.
- Refonte tarifaire (coût par feature, économie d'unités d'usage) — futur document de tarification.
- Spec opérationnelle de l'API entre instance et service (auth, endpoints, schémas) — futur `infrastructure.md`.
