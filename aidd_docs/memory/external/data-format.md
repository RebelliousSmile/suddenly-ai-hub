# Format des données — rows de tables

> Spécifie le format physique des rows persistées dans les tables de Muses. Découle de `architecture-tables-ml.md` (niveaux de granularité, tagging axial) et `learning-and-trust.md` (provenance, quality score). En cas de conflit, ces deux docs font foi.

## Format conteneur

- **1 fichier JSONL par table** — une row par ligne, JSON unique par ligne.
- Versionnés en git (diff-friendly, audit trivial).
- Nommage : `tables/<niveau>/<slot>__<tags>.jsonl` (ex : `tables/fragments/dialogue__combat__hostile__narquois.jsonl`).
- Un index SQLite avec FTS5 sur les colonnes de tags accélère les requêtes — reconstruit depuis les JSONL.
- Embeddings pré-calculés des rows : fichiers `.npy` adjacents à chaque JSONL (`<table>.embeddings.npy`).

## Schéma commun à toutes les rows

Champs communs aux quatre niveaux de granularité (entités, templates, beats, fragments) :

| Champ              | Type      | Requis | Description                                                              |
| ------------------ | --------- | ------ | ------------------------------------------------------------------------ |
| `id`               | UUID      | oui    | Identifiant interne Muses                                                |
| `level`            | enum      | oui    | `entity` \| `template` \| `beat` \| `fragment`                           |
| `tags`             | object    | oui    | Tags sur les cinq axes canoniques (cf. ci-dessous)                       |
| `content`          | varie     | oui    | Contenu utile (string ou objet structuré, dépend de `level`)             |
| `user_id`          | string    | oui*   | Acteur ActivityPub (URI) ; `null` pour `source: bootstrap` ou `mined`    |
| `instance_id`      | string    | oui*   | Domaine de l'instance source ; `null` pour `source: bootstrap`           |
| `created_at`       | timestamp | oui    | ISO 8601 UTC, date d'ingestion côté Muses                                |
| `source`           | enum      | oui    | `bootstrap` \| `contribution_explicit` \| `derived_from_edit` \| `mined` |
| `signature`        | string    | oui*   | Signature HTTP ActivityPub de la soumission (sauf `bootstrap`, `mined`)  |
| `quality_score`    | float     | non    | Score courant calculé par le quality gating (cf. `learning-and-trust.md` §7) ; absent à l'ingestion |
| `archived_at`      | timestamp | non    | Date d'archivage si row sortie du service actif ; absent sinon           |

\* obligatoire sauf cas explicitement exemptés par la valeur de `source`.

## Format des tags axiaux

`tags` est un objet à cinq clés correspondant aux cinq axes canoniques (`philosophy.md` § Conventions). Chaque clé porte une **liste** de valeurs — une row peut être valide dans plusieurs valeurs d'un même axe (ex : `combat` ET `intrigue`).

```json
{
  "tags": {
    "univers": ["medieval-fantastique"],
    "situation": ["combat"],
    "rapport_initial": ["hostile", "neutre"],
    "voix": ["narquois"],
    "emotion_dominante": ["colere", "peur"]
  }
}
```

Une liste vide ou clé absente signifie « universel sur cet axe » — la row matche n'importe quelle valeur. À utiliser avec parcimonie ; la plupart des rows ont des tags spécifiques.

## Champ `content` selon le niveau

### `level: entity`

Unité lexicale typée (geste, émotion, lieu, objet, nom propre, trait…).

```json
{
  "content": {
    "type": "geste",
    "lemma": "serrer les poings",
    "variants": {
      "genre": ["m", "f"],
      "nombre": ["s", "p"],
      "tense": ["present", "passe-compose", "imparfait"]
    },
    "forms": {
      "m.s.present": "serre les poings",
      "f.s.present": "serre les poings",
      "m.p.present": "serrent les poings",
      "m.s.passe-compose": "a serré les poings"
    }
  }
}
```

`type` est typé (geste, emotion, lieu, objet, nom_personne, nom_lieu, trait, gestation…). La liste fermée des `type` est dans `axes-and-tags.md` (à venir).

### `level: template`

Squelette de phrase avec slots typés.

```json
{
  "content": {
    "skeleton": "{char.name} {action.geste} en {emotion}, {action.suite}",
    "slots": {
      "char.name": {"source": "context"},
      "action.geste": {"source": "table:entities", "type": "geste"},
      "emotion": {"source": "table:entities", "type": "emotion"},
      "action.suite": {"source": "table:fragments", "tags_match": true}
    }
  }
}
```

`source` indique où le remplissage tire :
- `context` — depuis le contexte de la requête (fiche perso, derniers reports).
- `table:<niveau>` — tirage dans une table de ce niveau.
- `static` — valeur fixe inscrite dans le template.

`tags_match: true` signifie que le slot hérite des tags de la row template courante pour son tirage.

### `level: beat`

Unité narrative de niveau scène.

```json
{
  "content": {
    "label": "hesitation",
    "description": "Le personnage hésite avant d'agir ou de parler.",
    "typical_templates": ["uuid-template-1", "uuid-template-2"],
    "arc_position": ["debut", "milieu"]
  }
}
```

`arc_position` : où dans l'arc de scène ce beat est typiquement joué (`debut`, `milieu`, `tournant`, `fin`). Vide = libre.

### `level: fragment`

Sortie complète prête à insérer.

```json
{
  "content": {
    "text": "« Tu rigoles, j'espère ? » lance-t-elle sans même lever les yeux.",
    "char_pov": "neutral",
    "beat_played": "raillerie"
  }
}
```

`char_pov` : indique si le fragment est neutre (`neutral`), à la première personne du POV joueur (`pov-player`), ou à la troisième (`third-person`).
`beat_played` : référence au beat narratif que ce fragment incarne (optionnel, sert au matching à l'étage 3).

## Exemple minimal d'un fichier table

`tables/fragments/dialogue__combat__hostile__narquois.jsonl` :

```json
{"id":"f47a...","level":"fragment","tags":{"univers":["medieval-fantastique"],"situation":["combat"],"rapport_initial":["hostile"],"voix":["narquois"],"emotion_dominante":["colere"]},"content":{"text":"« Voilà tout ce que tu as ? »","char_pov":"neutral","beat_played":"provocation"},"user_id":"https://exemple.tld/users/alice","instance_id":"exemple.tld","created_at":"2026-05-17T14:23:00Z","source":"contribution_explicit","signature":"keyId=\"...\",..."}
{"id":"f48b...","level":"fragment","tags":{"univers":["medieval-fantastique"],"situation":["combat"],"rapport_initial":["hostile"],"voix":["narquois"],"emotion_dominante":["peur"]},"content":{"text":"« Tu vas le regretter, crois-moi », murmure-t-il, la voix plus rauque qu'il ne l'aurait voulu.","char_pov":"third-person","beat_played":"menace-feinte"},"user_id":null,"instance_id":null,"created_at":"2026-05-10T00:00:00Z","source":"bootstrap","signature":null}
```

## Index SQLite

Construit et reconstruit depuis les JSONL — pas la source de vérité, juste une accélération de query.

```sql
CREATE TABLE rows (
    id TEXT PRIMARY KEY,
    table_path TEXT NOT NULL,
    level TEXT NOT NULL,
    user_id TEXT,
    instance_id TEXT,
    source TEXT NOT NULL,
    created_at TEXT NOT NULL,
    quality_score REAL,
    archived_at TEXT,
    -- tags dénormalisés pour query rapide
    tag_univers TEXT,            -- json array
    tag_situation TEXT,
    tag_rapport_initial TEXT,
    tag_voix TEXT,
    tag_emotion_dominante TEXT,
    content_text TEXT            -- pour FTS5
);

CREATE VIRTUAL TABLE rows_fts USING fts5(content_text, content=rows);
```

## Embeddings

Une matrice `numpy` par fichier table, dimension `(n_rows, embedding_dim)`, ligne `i` = embedding de la row `i` du JSONL. Recalculée à l'ingestion. Modèle d'embedding fixé dans `infrastructure.md` (à venir) — typiquement un sentence-transformer multilingue CPU-friendly.

Le mapping `id` ↔ index de ligne est porté par le JSONL lui-même (ordre des lignes = ordre des embeddings).

## Validation à l'ingestion

Avant qu'une row soit acceptée dans une table :

1. Schéma : champs requis présents, types corrects, `level` cohérent avec `content`.
2. Signature ActivityPub vérifiée contre la clé publique de l'instance source (sauf `bootstrap`, `mined`).
3. Anonymisation : `content` ne contient pas d'informations personnelles identifiables résiduelles (run du pipeline `pipelines/anonymization/`).
4. Tags : valeurs des cinq axes appartiennent au set canonique de `axes-and-tags.md` (à venir).
5. Trust : seuil minimum atteint pour `user_id`, sauf si `source` dispense.

Les rows rejetées sont logguées avec motif. Pas d'erreur silencieuse.

## Hors périmètre de ce document

- Taxonomie détaillée des valeurs par axe — futur `axes-and-tags.md`.
- Modèle d'embedding précis — futur `infrastructure.md`.
- Algorithme exact de validation d'anonymisation — `pipelines/anonymization/`.
- Format des **patterns** pour le pipeline d'analyse (tables-patterns pour features #81-#84) — futur `analysis-pipeline.md`.
