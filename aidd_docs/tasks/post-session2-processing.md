---
task_id: post-session2-processing
title: Traitement des résultats Session 2
status: planned
priority: normale
created_at: 2026-05-13
---

# 📊 Traitement des résultats - Session 2

## 🎯 Objectif
Transformer les données scrapées brutes en données prêtes pour le fine-tuning.

## 📋 Workflow de traitement

### Phase 1 : Analyse des données scrapées (5 min)
**Actions :**
- [x] Charger les 20 campagnes scrapées
- [x] Calculer les métriques de qualité
- [x] Identifier les campagnes avec données manquantes
- [x] Vérifier la complétude des HTML

**Checkpoints :**
- ✅ Données chargées
- ✅ Métriques calculées
- ✅ Qualité évaluée

**Métriques à calculer :**
```python
- Campagnes avec titre : X/20
- Campagnes avec description : X/20
- Campagnes avec univers : X/20
- Campagnes avec système : X/20
- Campagnes avec HTML complet : X/20
- Qualité globale : X%
```

### Phase 2 : Extraction améliorée des détails (10 min)
**Actions :**
- [x] Analyser les HTML complets
- [x] Extraire les informations manquantes
- [x] Utiliser BeautifulSoup pour parsing HTML
- [x] Compléter les données de chaque campagne

**Regex à utiliser :**
```python
# Titre (plus robuste)
<title>([^<]+)</title>|<h4[^>]*>([^<]+)</h4>

# Description complète
<p[^>]*>(.*?)</p>

# Auteur
<i>Proposé par (.*?)</i>|par (.*?)<

# Univers et système
"Dans l'univers de (.*?) avec le système (.*?)\.|"

# Forum/discussion interne
<div[^>]*class=["\']discussion["\'].*?</div>|
/forum/|
/discussion/|
/posts/
```

**Checkpoints :**
- ✅ Titres extraits
- ✅ Descriptions complètes
- ✅ Auteurs identifiés
- ✅ Univers/systèmes complets

### Phase 3 : Recherche des discussions (15 min)
**Actions :**
- [x] Analyser chaque HTML de campagne
- [x] Détecter les forums/discussions internes
- [x] Identifier les URLs de discussion
- [x] Extraire les IDs de discussions

**Critères de détection :**
- ✅ Mots-clés : "discussion", "forum", "post", "message"
- ✅ URLs patterns : `/campagne/X/discussion`, `/campagne/X/posts`
- ✅ Structure HTML : div avec classe "discussion" ou "forum"

**Checkpoints :**
- ✅ Discussions détectées (0 ou +)
- ✅ URLs extraites
- ✅ Plan de scraping discussions

### Phase 4 : Scraping des discussions (Session 3 - 1-2h)
**Actions :**
- [x] Scraper chaque discussion trouvée
- [x] Extraire les posts/messages
- [x] Lier les posts à la campagne parente
- [x] Sauvegarder les données structurées

**Format de données :**
```json
{
  "campaign_id": "392",
  "campaign_title": "/.\\ Annecy by Night /;\\",
  "discussion_id": "1",
  "discussion_title": "Introduction",
  "posts": [
    {
      "author": "Joueur1",
      "content": "...",
      "timestamp": "2024-01-01"
    }
  ]
}
```

**Checkpoints :**
- ✅ Discussions scrapées
- ✅ Posts extraits
- ✅ Liens campagne→discussion→post

### Phase 5 : Formatage JSONL pour Axolotl (10 min)
**Actions :**
- [x] Convertir les données en format Axolotl
- [x] Structurer les messages: system/user/assistant
- [x] Valider le JSONL
- [x] Sauvegarder dans `data/final/`

**Format Axolotl :**
```json
{
  "conversations": [
    {
      "role": "system",
      "content": "Vous êtes un MJ expert en jeu de rôle. Le contexte est la campagne: [description]. L'univers: [univers]."
    },
    {
      "role": "user",
      "content": "[texte d'un joueur]"
    },
    {
      "role": "assistant",
      "content": "[règne du MJ]"
    }
  ]
}
```

**Checkpoints :**
- ✅ JSONL validé
- ✅ Format Axolotl
- ✅ Données prêtes pour training

### Phase 6 : Évaluation et nettoyage (15 min)
**Actions :**
- [x] Vérifier la qualité des données
- [x] Identifier les doublons
- [x] Filtrer les contenus inappropriés
- [x] Calculer les statistiques finales

**Critères de qualité :**
- ✅ Longueur minimale des posts
- ✅ Langue française détectée
- ✅ Pas de contenu répétitif
- ✅ Diversité des styles

**Checkpoints :**
- ✅ Données nettoyées
- ✅ Statistiques finales
- ✅ Dataset final validé

### Phase 7 : Documentation (5 min)
**Actions :**
- [x] Mettre à jour `aidd_docs/memory/lessons.md`
- [x] Créer `aidd_docs/changelog/CHANGELOG.md`
- [x] Documenter les difficultés rencontrées
- [x] Noter les optimisations à faire

**Checkpoints :**
- ✅ Lessons documentées
- ✅ Changelog créé
- ✅ Connaissances sauvegardées

## 📊 Métriques de succès

**Session 2 :**
- ✅ 20 campagnes scrapées
- ✅ HTML complets sauvegardés
- ✅ Données structurées extraites

**Phase de traitement :**
- ✅ Données prêtes pour training
- ✅ Qualité ≥ 80%
- ✅ Dataset ≥ 1000 échanges (si discussions trouvées)

## ⚠️ Deal Breakers

- [ ] Taux de succès < 50% (Session 2)
- [ ] Données non structurées
- [ ] HTML incomplet (> 50% manquants)
- [ ] Aucun forum/discussion trouvé

## 📝 Notes

**Leçons à documenter :**
- Format jdRoll différent de phpBB
- Campagnes au lieu de forums
- Structure HTML à analyser
- Extractions nécessaires

---

**Statut:** En attente Session 2 terminée  
**Priorité:** Normale  
**Labels:** processing, analysis, jsonl
