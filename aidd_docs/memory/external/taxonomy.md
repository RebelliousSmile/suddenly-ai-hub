# Taxonomie Genres et Situations RP

> **Source** : Issue #33 — [SPIKE] Définir la taxonomie genres et situations RP
> **Date** : 2026-05-15
> **Statut** : ✅ Décision actée

---

Cette taxonomie définit les deux axes de spécialisation pour les LoRA Suddenly.

## Axe 1 — Genres (Univers)

Basé sur la taxonomie du GROG (legrog.org/themes). Chaque genre donne lieu à un LoRA dédié.

| Genre | Description |
|-------|-------------|
| **Médiéval Fantastique** | Épopées, royaumes, magie, dragons. |
| **Historique Fantastique** | Période historique + magie/urnaturel. |
| **Science Fiction (Sci-Fi)** | Futur, space opera, dystopies. |
| **Contemporain Fantastique** | Monde moderne + surnaturel. |
| **Space Opera** | Conflits interstellaires, grands ensembles. |
| **Contemporain** | Monde réel moderne, sans fantastique. |
| **Post-Apocalyptique** | Monde après catastrophe, survie. |
| **Cyberpunk / Anticipation** | Futur proche, technologie, sombre. |
| **Super-héros** | Pouvoirs, organisations, justice. |
| **Oriental / Manga** | Thèmes asiatiques, styles manga. |
| **Merveilleux / Onirique** | Rêve, magie, absurde. |
| **Humoristique** | Comédie, parodie, léger. |
| **Univers parallèles** | Multivers, réalités alternatives. |
| **Générique** | Sessions sans genre identifié (fallback neutre). |

## Axe 2 — Situations (Thématiques)

Thématiques transversales qui influencent le style de réponse (actions, dialogues, descriptions).

| Situation | Description |
|-----------|-------------|
| **Romance / Relation** | Interactions sociales, sentiments, séduction. |
| **Combat / Bataille** | Actions physiques, stratégies, violence. |
| **Enquête / Investigation** | Mystères, indices, raisonnements. |
| **Diplomatie / Négociation** | Politiques, échanges verbaux, tractations. |
| **Exploration / Découverte** | Découverte de lieux, artefacts, inconnu. |
| **Introspection / Monologue** | Réflexion intérieure, sentiments profonds. |

## Seuil de déclenchement

- **500 sessions** par catégorie avant de lancer un entraînement LoRA dédié.

## Mécanisme de tagging

- L'instance Suddenly doit taguer chaque session avec un (genre + situation).
- Le mécanisme peut être libre, listes fermées, ou suggestion automatique.
