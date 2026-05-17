---
name: learning-and-trust
description: Régime de données bootstrap puis continu, online learning des étages ML, trust contextuel et résilience décentralisée
---

# Apprentissage continu et trust

> Ancre dans `philosophy.md` §2 (« Une couche, pas une instance »), §3 (« Continu, pas batch ») et §5 (« Décentralisé et responsabilisé »). Spécifie comment Muses ingère des données et apprend en continu sans re-training batch, et comment il filtre les contributions ouvertes du Fediverse sans modération réactive 24/7.

Le pipeline 4-étages et les niveaux de tables auxquels ce document fait référence sont définis dans `architecture-tables-ml.md`. Le profil de style auteur (orthogonal au trust) est spécifié dans `style-coaching.md`.

## 1. Régime de données : bootstrap → continu

Muses connaît deux régimes successifs, sans cut-off net entre les deux — déplacement progressif du centre de gravité.

### Phase bootstrap (one-shot)

- Sources : corpus de visual novels, romans, scrapings de forums RP existants (`pipelines/anonymization`, `pipelines/crawl_rpv`).
- Pipeline : extraction NER + segmentation en beats + clustering sémantique → tables candidates → curation manuelle → premières tables publiées.
- Volume cible : suffisant pour ne pas servir de tables vides aux premiers contributeurs. Pas pour atteindre la qualité finale.
- Toutes les rows issues du bootstrap portent `source: bootstrap` dans leur provenance.

### Régime nominal (continu)

- Sources : contributions des joueurs des instances Suddenly connectées, captées via les signaux UI (cf. `style-coaching.md` §3) et les soumissions explicites opt-in.
- Pipeline : chaque accept/édition génère un signal qui met à jour les composants ML ; chaque édition originale ou contribution explicite produit un fragment candidat ajouté à la table appropriée.
- Aucun re-training batch. Aucune version de modèle.

### Transition

Le système ne « bascule » jamais formellement de bootstrap à nominal. La phase nominale grandit en parallèle ; les rows bootstrap restent en place tant qu'elles ne sont pas archivées par le quality gating (§5). À terme, la proportion de rows `source: bootstrap` décroît mécaniquement par dilution.

## 2. Provenance — schéma de row obligatoire

Chaque row de chaque table porte une provenance complète. Pas d'exception, y compris pour les rows bootstrap (provenance synthétique).

| Champ              | Type      | Description                                                         |
| ------------------ | --------- | ------------------------------------------------------------------- |
| `id`               | UUID      | Identifiant interne Muses                                           |
| `user_id`          | string    | Acteur ActivityPub (URI), `null` pour `source: bootstrap`           |
| `instance_id`      | string    | Domaine de l'instance source                                        |
| `created_at`       | timestamp | Date d'ingestion côté Muses                                         |
| `source`           | enum      | `bootstrap` \| `contribution_explicit` \| `derived_from_edit` \| `mined` (la valeur `derived_from_edit` est produite par le signal `accept_edited` décrit dans `style-coaching.md` §3) |
| `signature`        | string    | Signature ActivityPub HTTP de la soumission (sauf `bootstrap` et `mined`) |
| `quality_score`    | float     | Score courant calculé par le quality gating, mis à jour en continu  |
| `created_for_axis` | dict      | Axes contextuels de la session d'origine (snapshot)                 |

La provenance permet : attribution, pondération par trust user et trust instance, archivage ciblé, audit.

## 3. Online learning des étages ML

Aucun entraînement batch. Trois étages sur quatre apprennent en continu.

### Étage 1 — Sélecteur

- **Forme** : classifieur multi-label logistique sur embedding du contexte → P(table_t pertinente | ctx).
- **Update** : SGD online à chaque session terminée. Signal positif sur les tables d'où viennent les rows acceptées ; signal négatif sur les tables des rows ignorées.
- **Snapshots** : poids sérialisés toutes les N heures pour rollback en cas de drift constaté.

### Étage 2 — Pondérateur

- **Forme** : similarité contexte ↔ row dans l'espace d'embedding partagé, soft-maxée sur les rows de chaque table sélectionnée.
- **Update** : pairwise learning. Sur un couple `(accept row r, reject_off row r')` dans le même contexte, gradient hinge : pousse `sim(ctx, r) > sim(ctx, r')`.
- L'étage 2 est aussi le point d'application de la pondération trust et de la modulation confort/challenge (cf. `style-coaching.md` §2). Ces multiplicateurs s'appliquent **après** la distribution apprise.

### Étage 3 — Recombinateur

- **N'apprend pas.** Règles d'assemblage déterministes + remplissage de slots typés.
- Les variantes d'accord (genre, nombre, temps) sont pré-stockées dans les rows, pas générées.

### Étage 4 — Filtreur

- **Forme** : scoreur best-of-N (cross-encoder léger ou somme pondérée de features) sur `(contexte, candidat)`.
- **Update** : preference learning style DPO-lite. Sur un signal `(winner candidat retenu, loser candidat dominé)`, gradient pousse `s(winner) - s(loser)` vers le positif.
- **Désambiguïsation** : les signaux `reject_challenge_appreciated` (cf. `style-coaching.md` §3) **n'alimentent pas** l'update du filtreur — sinon le mode challenge collapse.

### Snapshots et rollback

Les poids des étages 1/2/4 sont snapshottés périodiquement avec horodatage. En cas de dérive constatée (drift, raid massif détecté a posteriori, bug d'update), rollback possible à un snapshot daté. Les rows ajoutées entre-temps **ne sont pas** rétro-supprimées, mais leurs signaux d'apprentissage le sont.

## 4. Trust contextuel par auteur

Vecteur de Beta reputations indexé par `(axe, valeur)` — cf. message précédent dans la conversation et `philosophy.md` §5.

```
trust[user_id][axis][value] = (alpha, beta, last_update_ts)
```

### Update

Symétrique avec la table à 5 signaux de `style-coaching.md` §3 (colonne *Update trust contributeur*) :

- **Accept** par un tiers sur une row contribuée par l'auteur dans contexte `(axis, value)` : `α += w_accept`.
- **Accept_edited** : `α += w_accept_edited` (atténué — le contenu de l'auteur était proche mais pas suffisant).
- **Reject_off** : `β += w_reject`.
- **Reject_challenge_appreciated** : neutre. Le challenge ne pénalise pas son auteur.
- **Ignore** : `β += w_ignore` (faible).

Les poids `w_*` sont calibrés tels que `w_accept ≈ 1`, `w_accept_edited ≈ 0.5`, `w_reject ≈ 1`, `w_ignore ≈ 0.2`. À ajuster sur données réelles.

### Query

- **Trust moyen** : `α / (α + β)`.
- **Confiance** : `1 / variance(Beta(α, β))`. Distingue 95% sur 1000 contribs (haute confiance) de 95% sur 5 contribs (faible).
- En pondération downstream, on utilise un **trust pénalisé par la confiance** — un user à 95% sur 5 contribs vaut `0.5 + (0.95 - 0.5) × confidence_factor`, pas brutalement `0.95`.

### Décroissance temporelle

Demi-vie de **~6 mois** sur `α` et `β` (plus longue que la demi-vie ~3 mois du profil de style — un auteur peut évoluer en style sans devenir moins fiable). Décroissance appliquée au calcul (pas mutation in-place) pour permettre l'audit.

### Cold start

Prior `Beta(1, 1)` : trust = 0.5, confiance basse. Multiplié par la réputation d'instance (§5), ça donne un poids initial raisonnable.

## 5. Réputation d'instance

```
instance_weight[instance_id] = (
    base_multiplier: float ∈ [0.3, 1.2],
    reviewed_at: timestamp,
    source: 'auto' | 'admin'
)
```

### Calcul automatique

Agrégat sur les contribs de l'instance ces N derniers mois :
- taux de `reject_off` moyen,
- taux d'archivage par quality gating,
- volume / régularité,
- signalements inter-instances.

Le multiplicateur est borné `[0.3, 1.2]` — une mauvaise instance ne fait pas tomber ses users à 0, et une bonne instance ne crée pas d'oligarchie.

### Override admin

Admin Muses peut figer un `base_multiplier` (ex: instance fraîchement créée mais opérée par une communauté connue), avec `source: 'admin'`. La revue auto reprend ensuite si l'override expire.

### Application

Au moment de la pondération à l'étage 2, le multiplicateur d'instance porte sur le **trust effectif du user au moment de la contribution** :

```
weight_eff(row) =
      prob_apprise(row | ctx)
    × Π_axis  trust_penalized(row.user, axis, ctx[axis])
    × instance_weight[row.instance].base_multiplier
    × time_decay(row.created_at)
    × confort_challenge_modulation(row, user_courant)
```

## 6. Garde-fous décentralisés

### Anti-sleeper

Borner le **gain max** de `α` par fenêtre temporelle (proposé : `+10` par jour par `(axe, valeur)`). Un user qui ferait soudain 500 contribs acceptables en 24h ne saute pas à un trust ultra-confiant. La construction de réputation reste lente — c'est voulu.

### Anti-takeover

Détection d'**anomalie comportementale** sur les contribs : embedding moyen des K dernières contribs vs profil historique du user. Déviation supérieure à un seuil → suspension temporaire du bénéfice du trust + alerte admin Muses. L'auteur peut écrire normalement, ses contribs juste ne portent plus son ancien trust.

### Anti-sybil

- Seuil minimum d'activité (M contribs sur P semaines) avant que le trust commence à compter au-delà du prior.
- Couplage à la signature ActivityPub — un nouveau user dans une instance existante hérite partiellement de la réputation d'instance, mais doit construire son propre trust.
- Détection statistique de clusters de comptes corrélés (signature comportementale) — sans la résoudre côté Muses, mais en signalant aux admins de l'instance source.

### Anti-cold-start sur zones rares

Maintenir une **carte de couverture** sur l'hypercube canonique `(univers × situation × rapport_initial × voix × emotion_dominante)` — définie dans `architecture-tables-ml.md` § Carte de couverture contextuelle. Les cellules sous-peuplées sont marquées et priorisées pour le mining bootstrap futur. L'étage 1 peut tomber gracieusement sur des tables voisines (fallback hiérarchique sur axes, ordre `emotion_dominante → voix → rapport_initial → situation → univers`).

## 7. Quality gating

Score qualité maintenu par row, indépendant du trust de l'auteur.

```
quality_score[row] = f(
    accepts_in_similar_contexts,
    rejects_in_similar_contexts,
    edits_count,
    age,
)
```

La fonction `f()` est délibérément laissée non spécifiée à ce stade — elle sera calibrée empiriquement en POC sur les premières données de production (forme candidate : moyenne pondérée des taux d'accept par contexte, pénalisée par le ratio d'éditions et amortie sur l'âge). Le doc sera figé une fois la calibration tenable.

- **Archivage** des rows sous seuil après période d'observation (pas suppression — archivage permet audit et restauration).
- **A/B online** : pour des contextes équivalents, deux rows candidates peuvent être servies en alternance pour mesurer leur performance comparée et accélérer la convergence.
- **Promotion** : une row à très haut quality_score peut voir son embedding renforcé (poids accru à l'étage 2 indépendamment du trust auteur).

## 8. Hors périmètre / questions ouvertes

- **Politique de rétention RGPD** : conservation des rows après suppression de compte user. Probable anonymisation (drop du `user_id`, garde du `instance_id`), à confirmer juridiquement.
- **Continuité d'identité d'un contributeur** — trois cas à traiter ensemble dans un futur protocole d'identité :
  - migration inter-instance (un user change d'instance Suddenly, l'acteur ActivityPub change) ;
  - rotation de clé ActivityPub d'un user (signature change, identité logique stable) ;
  - takeover détecté (cf. §6) puis résolu — restauration partielle du trust ?
  La piste par défaut est de s'appuyer sur l'activité ActivityPub `Move` *et* sur un mécanisme de claim signé par l'ancienne clé, mais le protocole reste à spécifier.
- **Visibilité du trust par l'admin d'instance source** : l'admin Muses voit tout ; l'admin d'une instance voit-il les trusts de ses propres users ?
- **Mécanisme de contestation** : un user qui pense subir un trust injustement bas peut-il déclencher une revue admin ?
- **Politique de signal pour signal de modération externe** (block d'instance prononcé par une autre instance) — affecte-t-il `instance_weight` ?
- **Calibration fine** des constantes (demi-vie 6 mois, bornes multiplicateurs, poids `w_*`, seuils d'archivage, taille des fenêtres anti-sleeper) — à itérer sur données réelles, pas à figer maintenant.
