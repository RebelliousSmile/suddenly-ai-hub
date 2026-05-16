---
name: philosophy
description: Identité et principes directeurs du projet Muses — ce qu'il est, ce qu'il n'est pas, et pourquoi
---

# Philosophie — Muses

> Muses est une couche d'assistance créative pour le Fediverse Suddenly. Une muse au sens classique : elle provoque, elle ne sert pas.

Ce document fixe l'identité du projet. Les choix d'architecture, de modèle de données et d'implémentation découlent de ces principes — non l'inverse. En cas de conflit entre une décision technique et un principe ici, le principe l'emporte (ou il faut explicitement actualiser ce document).

## 1. Une muse, pas un serviteur

L'objectif n'est pas de remplacer l'auteur, ni d'écrire à sa place. Muses **interroge** le style de l'auteur et le **pousse hors de ses habitudes** quand il le souhaite.

Deux modes coexistent :

- **Confort** — les suggestions s'alignent au style établi de l'auteur, pour fluidifier la rédaction.
- **Challenge** — les suggestions s'écartent volontairement des habitudes, pour ouvrir des chemins inexplorés.

Ce mode dual est constitutif du produit, pas une option. Sans le mode challenge, Muses deviendrait un amplificateur d'habitudes et appauvrirait progressivement la production des auteurs.

## 2. Une couche, pas une instance

Muses n'est pas une instance Suddenly de plus. C'est un **service unique, mutualisé** entre toutes les instances Suddenly du Fediverse, qui s'y connectent via ActivityPub.

Conséquences :

- Données et apprentissages **partagés** entre instances : une bonne contribution d'une instance bénéficie aux auteurs d'une autre.
- Modération **commune** : un système de trust contextuel arbitre la qualité des contributions, modulé par la réputation de chaque instance source.
- Auteurialité **individuelle** : chaque contribution reste attribuée à son auteur (signature ActivityPub).

## 3. Continu, pas batch

Muses ne se ré-entraîne pas. Il n'y a pas de « version 2.0 » du modèle.

- Les tables s'enrichissent **ligne par ligne** au fil des contributions.
- Les composants ML se mettent à jour **incrémentalement** sur chaque signal (accept / reject / édition).
- Le corpus VN / romans / forums sert uniquement à **amorcer la pompe** au démarrage. Une fois le régime nominal atteint, les contributions joueurs deviennent la source dominante.

Cette propriété disqualifie les LoRA et le fine-tuning batché, et motive le choix d'une architecture tables + petits modèles ML.

## 4. Boucle bidirectionnelle

Muses apprend des contributeurs. Mais il fait aussi **apprendre** les contributeurs.

- Le système peut surfacer des observations sur le style de chaque auteur (« tu surutilises tel beat », « tu n'as jamais exploré tel champ lexical »).
- Le mode challenge confronte volontairement l'auteur à des choix qu'il n'aurait pas faits seul.
- L'objectif co-construit : faire évoluer collectivement la qualité du corpus *et* la qualité d'écriture des contributeurs.

## 5. Décentralisé et responsabilisé

Les contributions sont ouvertes à tous les joueurs des instances Suddenly connectées. Mais cette ouverture est **pondérée**, pas naïve.

- **Trust contextuel par auteur** sur chaque axe (univers, situation, voix) — un auteur peut être fiable sur certains contextes et neutre ailleurs.
- **Réputation d'instance** comme multiplicateur global — une instance bien modérée renforce ses auteurs, une instance lax dilue les leurs.
- **Visibilité réservée aux administrateurs** — les auteurs ne voient pas leur trust score, pour éviter le gaming.

L'objectif : faire en sorte qu'un raid de contributions toxiques soit dilué par la masse des contributions de qualité, sans nécessiter une modération réactive 24/7.

## 6. Lisible et traçable

Aucune génération boîte-noire. Chaque sortie de Muses est **traçable** jusqu'aux lignes de table tirées et aux scores qui les ont sélectionnées.

- Les tables sont curées explicitement, en JSONL versionné en git.
- Les choix des étages ML sont inspectables (tirage dans table X, ligne Y, scoré Z par contexte W).
- Un auteur peut comprendre **pourquoi** Muses lui a proposé telle suggestion — et la contester.

Cette propriété distingue Muses des LLM commerciaux opaques. Elle découle du choix tables + ML léger, pas d'une couche d'explicabilité ajoutée a posteriori.

## 7. Frugal

Pas de GPU, pas d'inférence LLM en production. Tous les composants ML tournent sur CPU : classifieurs légers, embeddings, rerankers.

Ce choix n'est pas une contrainte budgétaire : c'est une condition de **viabilité économique** pour un service partagé et gratuit pour les instances Suddenly. Et une condition d'**indépendance technologique** vis-à-vis des providers d'inférence commerciaux.

## 8. Ce que Muses n'est pas

- Pas un LLM, pas un chatbot.
- Pas un substitut à l'auteur : il propose, l'auteur arbitre.
- Pas une instance Suddenly de plus.
- Pas un assistant généraliste : son terrain est la rédaction de fiction narrative et dialoguée (les features de `use-cases.md`). Pas la productivité bureautique, pas la traduction, pas le code.
- Pas un système versionné. Pas de « v2 du modèle ». C'est un service continu.
- Pas une inférence payante par token. Le coût est dans l'orchestration et la curation, pas dans la génération.

---

## Documents techniques liés

- `architecture-tables-ml.md` — pipeline 4-étages tables + ML, et asymétrie génération / analyse (existe).
- `learning-and-trust.md` — bootstrap → continu, online learning des étages ML, trust contextuel, réputation d'instance (à écrire).
- `style-coaching.md` — profil de style auteur, modes confort / challenge, meta-suggestions sur le style (à écrire).
