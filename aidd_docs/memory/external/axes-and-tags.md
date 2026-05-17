# Axes contextuels canoniques et taxonomie des valeurs

> Spécifie en détail les cinq axes canoniques posés dans `philosophy.md` § Conventions. Source unique des valeurs autorisées pour tagger les rows (cf. `data-format.md`) et indexer le trust et le profil de style (cf. `learning-and-trust.md`, `style-coaching.md`).

Remplace `taxonomy.md` pré-pivot (2 axes pour LoRA).

## Principes

- **Cinq axes atomiques indépendants** — chaque axe est tagué séparément, pas de tuples.
- **Identifiants ASCII snake_case** sans accent — convention de stockage et de query. Les libellés humains affichables peuvent porter des accents et de la ponctuation, mais l'identifiant en table reste ASCII.
- **Liste fermée par axe** — toute nouvelle valeur passe par une revue (procédure §7) avant ingestion. Pas d'extension silencieuse.
- **Polyvalence** — une row peut porter plusieurs valeurs sur un même axe (cf. `data-format.md` § format des tags).

## 1. `univers`

Genre et lore. Détermine la palette d'entités lexicales et de motifs disponibles.

| Identifiant                  | Libellé affichable             | Description |
|------------------------------|--------------------------------|-------------|
| `medieval_fantastique`       | Médiéval Fantastique           | Épopées, royaumes, magie, créatures fantastiques |
| `historique_fantastique`     | Historique Fantastique         | Période historique réelle + surnaturel |
| `science_fiction`            | Science-Fiction                | Futur, technologie avancée, dystopies |
| `space_opera`                | Space Opera                    | Conflits interstellaires, vastes ensembles galactiques |
| `cyberpunk`                  | Cyberpunk                      | Futur proche, mégacorporations, implants, sombre |
| `steampunk`                  | Steampunk                      | Vapeur, engrenages, esthétique victorienne uchronique |
| `post_apocalyptique`         | Post-Apocalyptique             | Monde après catastrophe, survie, ruines |
| `contemporain_fantastique`   | Contemporain Fantastique       | Monde moderne + surnaturel |
| `contemporain`               | Contemporain                   | Monde réel moderne, sans fantastique |
| `horreur_gothique`           | Horreur Gothique               | Vampires, manoirs, atmosphère oppressante |
| `super_heros`                | Super-Héros                    | Pouvoirs, organisations, justice |
| `oriental`                   | Oriental / Manga               | Esthétiques et thèmes asiatiques |
| `merveilleux`                | Merveilleux / Onirique         | Rêve, magie diffuse, absurde |
| `humoristique`               | Humoristique                   | Comédie, parodie |
| `univers_paralleles`         | Univers Parallèles             | Multivers, réalités alternatives |
| `generique`                  | Générique                      | Fallback neutre, hors genre identifié |

## 2. `situation`

Type de scène. Détermine les beats et templates qui font sens.

| Identifiant      | Libellé affichable       | Description |
|------------------|--------------------------|-------------|
| `combat`         | Combat                   | Escarmouches, batailles, action physique |
| `romance`        | Romance                  | Sentiments, séduction, tension émotionnelle |
| `intrigue`       | Intrigue                 | Manœuvres, trahisons, mystères |
| `politique`      | Politique                | Négociations, alliances, enjeux sociaux |
| `enquete`        | Enquête                  | Investigation, indices, raisonnement |
| `exploration`    | Exploration              | Découverte de lieux ou d'artefacts |
| `introspection`  | Introspection            | Monologue intérieur, réflexion |
| `quotidien`      | Quotidien                | Moments de repos, interactions sociales légères |

## 3. `rapport_initial`

Relation entre les protagonistes au moment où commence la scène. Conditionne radicalement le ton.

| Identifiant   | Libellé affichable   | Description |
|---------------|----------------------|-------------|
| `hostile`     | Hostile              | Adversité ouverte, méfiance forte, intentions opposées |
| `neutre`      | Neutre               | Inconnu, indifférent, ni allié ni adversaire |
| `amical`      | Amical               | Confiance, complicité, intentions alignées |

Un combat amical (sparring, duel d'entraînement) et un combat hostile (bataille à mort) partagent le label `situation: combat` mais divergent radicalement sur `rapport_initial` — d'où la nécessité d'un axe dédié.

## 4. `voix`

Style narratif du MJ. Indépendant du genre, de la situation et de l'émotion.

| Identifiant   | Libellé affichable    | Description |
|---------------|-----------------------|-------------|
| `solennel`    | Solennel              | Grave, épique, mesuré |
| `narquois`    | Narquois              | Ironique, pince-sans-rire, espiègle |
| `theatral`    | Théâtral              | Spectaculaire, emphatique, dramatique |
| `neutre`      | Neutre                | Sobre, direct, non-intrusif |
| `lyrique`     | Lyrique               | Poétique, descriptif, sensuel |

## 5. `emotion_dominante`

Émotion principale de la scène. Taxonomie **Ekman 6** — choix pour sa simplicité, son ancrage scientifique et la stabilité de ses traductions linguistiques.

| Identifiant   | Libellé affichable   | Description (registre RP) |
|---------------|----------------------|---------------------------|
| `colere`      | Colère               | Rage, frustration intense, indignation |
| `degout`      | Dégoût               | Répulsion physique ou morale, mépris |
| `peur`        | Peur                 | Crainte, angoisse, terreur, anticipation négative |
| `joie`        | Joie                 | Bonheur, allégresse, fierté |
| `tristesse`   | Tristesse            | Mélancolie, deuil, abattement |
| `surprise`    | Surprise             | Étonnement, sidération |

L'absence d'`anticipation` et de `confiance` (présents dans la roue de Plutchik) est délibérée — Ekman 6 reste la base la plus partagée et la plus simple à annoter. Une extension vers Plutchik 8 pourra être envisagée plus tard si les données montrent que la nuance manque, mais c'est hors périmètre actuel.

## 6. Mécanisme de tagging

### Côté contribution

Une row produite par un user, ou une session source, porte des tags sur tout ou partie des cinq axes. Le tagging peut être :

- **Explicite** — l'auteur sélectionne dans des listes fermées au moment de la contribution.
- **Hérité** — la row prend les tags de la session ou du report source.
- **Inféré** — un classifieur léger propose des tags sur les axes non explicitement taggués, l'auteur peut corriger.

### Côté requête

Quand le service Muses sert une suggestion, le contexte de la session fournit les cinq tags axiaux (mêmes mécanismes : explicite, hérité, inféré). Le sélecteur (étage 1) filtre les rows par tags compatibles avant tirage.

### Compatibilité

Une row est **compatible** avec un contexte sur un axe si :

- la row n'a pas de tag sur cet axe (universelle), ou
- l'intersection entre les valeurs de la row et la valeur du contexte est non vide.

Compatibilité globale = ET sur les cinq axes.

## 7. Procédure d'ajout d'une valeur

Toute extension de la liste fermée d'un axe passe par :

1. Proposition explicite (PR ou ticket) avec justification : exemple concret de row qui ne rentre dans aucune valeur existante, et qui ne devrait pas être forcée dans la valeur la plus proche.
2. Revue : impact sur la carte de couverture (création d'une nouvelle dimension de cellule, donc creusement potentiel des données existantes).
3. Canonisation : ajout au présent document avec un identifiant ASCII snake_case et un libellé affichable.
4. Migration : les rows existantes ne sont **pas** rétro-taggées automatiquement. Elles peuvent l'être à la marge par re-classement.

Les valeurs ne sont **pas** retirées rétroactivement — au pire elles sont marquées « dépréciée » avec une recommandation de redirection.

## 8. Hors périmètre

- **Hiérarchie ou parenté entre valeurs** d'un axe (ex : `cyberpunk` est-il une spécialisation de `science_fiction` ?) — non pour l'instant ; chaque valeur est plate. Une éventuelle hiérarchie viendrait alourdir le tagging sans gain démontré.
- **Tagging multilingue** — pour l'instant FR. EN à envisager si la base de contributeurs devient bilingue.
- **Tagging automatique du corpus bootstrap** — relève des pipelines `pipelines/anonymization/` et `pipelines/crawl_rpv/`, pas de ce document.
