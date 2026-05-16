---
name: style-coaching
description: Profil de style auteur, modes confort et challenge, méta-suggestions sur le style
---

# Style coaching — Profil auteur et modes confort / challenge

> Ancre dans `philosophy.md` §1 (« Une muse, pas un serviteur ») et §4 (« Boucle bidirectionnelle »). Spécifie le mécanisme par lequel Muses challenge un auteur sans dériver vers l'amplificateur d'habitudes que la seule boucle accept/reject produirait.

Le pipeline 4-étages auquel ce document fait référence est défini dans `architecture-tables-ml.md`.

## 1. Profil de style auteur

Le profil de style est distinct du trust, sur un axe orthogonal.

|                    | Trust                                              | Profil de style                                                |
| ------------------ | -------------------------------------------------- | -------------------------------------------------------------- |
| Question répondue  | Cet auteur est-il fiable comme contributeur ?      | Cet auteur écrit dans quel style ?                             |
| Sources            | Accept/reject **des autres** sur ses contribs      | Accept/édition **par lui** des suggestions reçues + contenu rédigé |
| Forme              | Vecteur `(α, β)` par `(axe, valeur)`               | Histogrammes avec décroissance temporelle, à 4 niveaux         |
| Usage downstream   | Pondère ses contribs entrantes (étage 2)           | Personnalise les sorties qu'on lui propose (étages 2/3/4)      |

### Quatre niveaux d'observation

Parallèles aux quatre niveaux de tables :

- **Fréquence par row** — quelles lignes spécifiques l'auteur a acceptées ou réutilisées (histogramme sparse).
- **Fréquence par template** — quels squelettes il utilise spontanément.
- **Fréquence par beat** — quels beats narratifs il sollicite (`hésitation` dominante ? `rebondissement` rare ?).
- **Profil lexical** — vocabulaire dominant, longueur moyenne par feature, ratio dialogue / description / action / pensée.

Chaque dimension porte une **décroissance temporelle** (demi-vie ~3 mois proposée) — le profil reflète l'auteur courant, pas son passé fossilisé.

### Granularité contextuelle

Le profil est sous-indexé par `(axe, valeur)` au même niveau que le trust. Un auteur peut avoir un profil dense en `combat`+`medieval-fantastique` et quasi-vide en `romance`+`cyberpunk` — les deux situations sont traitées différemment par le système.

## 2. Deux modes opératoires

Le mode actif sur une requête est déterminé par :

- toggle UI explicite (« mode challenge » sur la session ou par requête) ;
- détection automatique (l'auteur réutilise N fois le même beat consécutivement → propose un challenge) ;
- feature dédiée (`propose-moi hors de ma zone`).

### Mode confort

Objectif : fluidifier la rédaction sans rompre le style.

| Étage             | Comportement confort                                                |
| ----------------- | ------------------------------------------------------------------- |
| 1. Sélecteur      | Inchangé — choix des tables par contexte                            |
| 2. Pondérateur    | **Bonus** sur les rows proches du profil (similarité haute → poids ↑) |
| 3. Recombinateur  | Favorise combos beats/templates **présents** dans l'historique récent |
| 4. Filtreur       | Best-of-N par cohérence contexte × proximité au profil              |

### Mode challenge

Objectif : sortir l'auteur de ses ornières, sans le sortir de son univers.

| Étage             | Comportement challenge                                              |
| ----------------- | ------------------------------------------------------------------- |
| 1. Sélecteur      | Peut inclure des tables sous-représentées dans son historique       |
| 2. Pondérateur    | **Malus** sur les rows trop proches du profil (similarité haute → poids ↓) |
| 3. Recombinateur  | Favorise combos beats/templates **absents** du profil récent        |
| 4. Filtreur       | Best-of-N avec **Maximum Marginal Relevance** contre le profil      |

Le malus de l'étage 2 reste **borné** : on diversifie l'éventail, on ne le retourne pas. L'auteur doit reconnaître son univers vu sous un angle différent, pas un univers étranger.

## 3. Cinq signaux utilisateur — l'instrumentation critique

Sans désambiguïsation des signaux, le ranker (étage 4) apprend à éviter les challenges (taux d'accept plus bas) et le mode challenge meurt en quelques semaines. C'est mathématique.

L'UI doit donc exposer **cinq** signaux distincts, pas deux :

| Signal                          | Sémantique                                          | Update profil de style                                       | Update ranker étage 4                            |
| ------------------------------- | --------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------ |
| `accept`                        | Inséré tel quel                                     | +1 sur row, template, beat, n-grammes                        | +1 sur `(contexte, row)`                         |
| `accept_edited`                 | Inséré avec édition                                 | +1 sur row brute ; édition capturée comme fragment candidat  | +1 atténué sur la row                            |
| `reject_off`                    | « Pas pertinent ici »                               | rien                                                         | -1 sur `(contexte, row)`                         |
| `reject_challenge_appreciated`  | « Bonne idée, pas pour cette scène »                | +1 *exploration* sur le beat/template du candidat            | **neutre** — pas de malus ranker                 |
| `ignore`                        | Bouton non cliqué après N secondes                  | rien                                                         | -0.2 sur `(contexte, row)`                       |

Le signal `reject_challenge_appreciated` est l'élément crucial. Il décorrèle « la suggestion était pertinente comme exploration » de « je l'ai insérée ». Sans ce signal, le système collapse vers le mode confort en quelques semaines.

Implication UI côté instance Suddenly : sous chaque suggestion, **deux boutons de rejet** — « pas ici » et « pas mal mais pas maintenant ». Pas un seul.

## 4. Méta-suggestions — un nouveau type de sortie

Sortie distincte des suggestions de texte. Format court, surfacé hors flux d'écriture (typiquement dans le profil utilisateur côté instance, ou en surface périodique).

Trois familles :

- **Observation sur l'usage** — « Tu as utilisé le beat *hésitation* dans 70% de tes scènes émotionnelles ce mois-ci. »
- **Suggestion d'exploration** — « Tu n'as jamais utilisé le champ lexical *organique* dans tes scènes cyberpunk. Tester ? »
- **Anti-pattern détecté** — « Tes descriptions de combat font en moyenne 40 mots quand tes autres descriptions en font 120. Volontaire ? »

Déclenchement : calcul périodique (batch nocturne) sur le profil de style. **Pas d'interruption dans le flux d'écriture.** Ces méta-suggestions ne consomment aucun Muses — décision à acter dans le document de tarification à venir.

## 5. Garde-fous

- **Calibration du malus challenge** — le multiplicateur étage 2 reste borné dans `[0.3, 1.0]`. Trop bas → l'auteur reçoit du off-style et abandonne le mode.
- **Cold start** — mode challenge désactivé tant que le profil n'a pas franchi un seuil minimal d'interactions (proposé : ~50 sur l'axe concerné). Sinon on challenge contre du vide.
- **Réversibilité** — l'auteur peut purger son profil de style à tout moment. RGPD-friendly et utile en cas de changement délibéré de direction stylistique.
- **Mode forcé hors challenge** — un auteur peut désactiver définitivement le mode challenge. Le produit reste pleinement utile en mode confort pur. Le challenge est une option, pas une imposition.
- **Visibilité du profil** — l'auteur voit son propre profil de style (au moins les méta-suggestions). C'est l'objet du coaching : sans visibilité, pas de prise de conscience.

Note : la visibilité du **profil de style** par l'auteur diffère de la visibilité du **trust** (admin-only — cf. `learning-and-trust.md` à venir). Le profil parle d'écriture, pas de fiabilité — partager le premier ne crée pas le risque de gaming associé au second.

## 6. Hors périmètre / questions ouvertes

- **Détection automatique du moment opportun** pour proposer un challenge. Heuristique à itérer en POC.
- **Profil par campagne / par groupe** — un auteur peut écrire différemment dans deux campagnes simultanées. Profil unifié, ou sous-vecteur indexé par `campaign_id` ?
- **Méta-suggestions sur le rythme** (longueur de scène, alternance dialogue/action/description/pensée) — pertinent ou hors scope ?
- **Calibration de la décroissance temporelle** du profil (3 mois proposé). À ajuster sur données réelles.
- **Coaching collectif** — méta-observations sur une campagne entière (« cette campagne a sur-représenté le beat *trahison* ces 2 mois ») — pertinent ou intrusif ?
