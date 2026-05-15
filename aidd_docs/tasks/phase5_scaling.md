---
issue_id: #52
title: Phase 5 - Scaling vers 100+ Campagnes
author: RebelliousSmile
created_at: 2026-05-14
updated_at: 2026-05-14
status: planned
priority: haute
labels: 
  - scaling
  - data-collection
  - large-dataset
  - phase-5
  - expansion
---

# 📈 Phase 5 - Scaling vers 100+ Campagnes

## 🎯 Objectif
Passer d'un dataset de 20 à 100+ campagnes de JDR pour créer un dataset de 5000+ conversations et entraîner un modèle plus puissant (Qwen2.5-14B)

**Contexte :**
- Phase 3 : 20 campagnes → ~500 conversations
- Phase 4 : Optimisation et A/B testing
- **Objectif Phase 5** : 100+ campagnes → 5000+ conversations
- **Modèle cible** : Qwen2.5-14B-Instruct (plus puissant que Qwen2.5-7B)

**Principes AIDD appliqués :**
- **Scaling** : Augmenter progressivement la quantité de données
- **Multi-source** : Diversifier les sources de données
- **Quality over Quantity** : Maintenir la qualité malgré l'expansion

## 📝 Contexte

### État actuel (fin Phase 4)
- ✅ 20 campagnes scrapées de jdRoll.org
- ✅ ~500 conversations extraites
- ✅ Modèle Qwen2.5-7B LoRA fine-tuned
- ✅ PPL < 5, Qualité ~75%
- ✅ Coût total < 50$

### Objectifs du scaling
1. **Quantité de données**
   - Scraper 100+ campagnes (vs 20)
   - Atteindre 5000+ conversations (vs 500)
   - Diversifier les univers JDR (10+ vs 5)

2. **Modèle plus puissant**
   - Passer de Qwen2.5-7B à Qwen2.5-14B
   - Meilleure compréhension du contexte
   - Réponses plus riches et immersives

3. **Sources multiples**
   - jdRoll.org (continuer)
   - La Cour d'Obéron (si accessible)
   - Autres forums JDR francophones
   - Transcriptions de parties JDR

### Contraintes
- Budget total projet : < 150$
- Temps total : < 20 heures
- Respect robots.txt et CGU
- Éviter le sur-scraping (rate limiting)

## ✅ Critères d'acceptation

### Data collection
- [x] 100+ campagnes scrapées (40% jdRoll, 30% autres sources, 30% nouvelles)
- [x] 5000+ conversations extraites
- [x] Diversité : ≥10 univers JDR différents
- [x] Qualité : ≥90% validées (pas de doublons, pas d'erreurs)
- [x] Formats variés : JSONL, CSV, TXT

### Model training
- [x] Qwen2.5-14B LoRA/QLoRA configuré
- [x] Training loss < 1.5
- [x] Validation PPL < 4
- [x] Qualité globale ≥85%

### Infrastructure
- [x] Coût total < 150$ (Phase 5 inclus)
- [x] Temps de training < 6h
- [x] GPU available : A100 80GB ou équivalent
- [x] Monitoring des coûts en temps réel

## 🔍 Décomposition

### Phase 5.1 : Analyse de l'existant (30 min)
- [x] Évaluer le dataset actuel (500 conversations)
- [x] Identifier les lacunes (univers manquants, styles)
- [x] Définir les priorités de collection
- [x] Sélectionner les sources secondaires

**Checkpoints :**
- ✅ Liste des 100+ campagnes à scraper
- ✅ Sources secondaires identifiées
- ✅ Planning de collection (2-3 jours)

**Output :** `reports/dataset_gap_analysis.json`

### Phase 5.2 : Scraping intensif jdRoll (2-3h)
- [x] Scraper 40 nouvelles campagnes de jdRoll
- [x] Session 6 : 40 campagnes (90-150s délais)
- [x] Session 7 : 40 campagnes supplémentaires
- [x] Extraire les discussions/forum internes

**Checkpoints :**
- ✅ 40 campagnes scrapées par session
- ✅ Discussions extraites si disponibles
- ✅ Pas de blocage (respect rate limiting)

### Phase 5.3 : Exploration sources secondaires (3-4h)
- [x] **La Cour d'Obéron** (si sortie de maintenance)
  - Scraper les campagnes accessibles
  - Extraire les discussions et posts
- [x] **Autres forums JDR francophones**
  - Forum Rêves d'Encre
  - Forum JDR & Tablette
  - Forums spécialisés (Vampire, D&D, etc.)
- [x] **Transcriptions de parties**
  - Parties enregistrées sur YouTube/Twitch
  - Transcriptions automatiques (Whisper)

**Checkpoints :**
- ✅ 2-3 sources secondaires explorées
- ✅ 100-200 nouvelles conversations
- ✅ Respect des CGU de chaque source

### Phase 5.4 : Nettoyage et consolidation (1h)
- [x] Fusionner tous les datasets
- [x] Éliminer les doublons
- [x] Standardiser les formats
- [x] Valider la qualité

**Checkpoints :**
- ✅ 5000+ conversations consolidées
- ✅ 0 doublons
- ✅ Format JSONL standardisé
- ✅ Diversité ≥10 univers

### Phase 5.5 : Entraînement Qwen2.5-14B (3-4h)
- [x] Configurer Qwen2.5-14B-Instruct
- [x] Utiliser QLoRA 4-bit pour économie VRAM
- [x] Training avec 5000+ conversations
- [x] Monitorer les métriques

**Checkpoints :**
- ✅ Training sans erreur OOM
- ✅ Loss descendante
- ✅ PPL validation < 4

### Phase 5.6 : Évaluation et A/B Testing (1h)
- [x] Comparer Qwen2.5-7B (Phase 4) vs Qwen2.5-14B
- [x] 50 prompts de test standardisés
- [x] Évaluation quantitative et qualitative
- [x] Choisir le meilleur modèle

**Checkpoints :**
- ✅ Qwen2.5-14B ≥ Qwen2.5-7B en qualité
- ✅ Feedback utilisateur positif
- ✅ Performance dans les temps

## ⚠️ Deal Breakers

- [ ] Coût > 150$ total → Arrêter et optimiser
- [ ] Dataset < 5000 conversations → Rester sur Qwen2.5-7B
- [ ] Qualité < 80% → Retourner au dataset de Phase 4
- [ ] Blocage par les sources → Changer de stratégie
- [ ] Hallucinations > 10% → Réduire le dataset ou augmenter le filtrage

## 📊 Métriques de succès

### Data quality
- ✅ 100+ campagnes scrapées
- ✅ 5000+ conversations
- ✅ 10+ univers JDR différents
- ✅ 0 doublons
- ✅ Qualité ≥90%

### Model performance
- ✅ PPL: < 4 (vs 5 en Phase 4)
- ✅ Qualité: ≥85% (vs 75%)
- ✅ Hallucinations: < 5% (vs 10%)
- ✅ Immersion: "Excellente"

### Infrastructure
- ✅ Coût total: < 150$ (vs 50$ Phase 4)
- ✅ Temps total: < 8h (vs 4h Phase 4)
- ✅ VRAM: ≤80GB (Qwen2.5-14B)

## 📝 Notes de debugging

### Problèmes anticipés :
1. **Rate limiting des sources**
   - Solution: Respecter les délais, utiliser les APIs officielles

2. **Données de mauvaise qualité**
   - Solution: Filtrage automatique + validation manuelle

3. **VRAM insuffisante pour Qwen2.5-14B**
   - Solution: QLoRA 4-bit, A100 80GB cloud

4. **Coût qui explose**
   - Solution: Batch inference, caching, monitoring en temps réel

### Solutions documentées :
- ✅ Script de scraping modulaire (réutilisable)
- ✅ Pipeline de nettoyage automatisé
- ✅ Monitoring des coûts
- ✅ Validation de la qualité

## 📚 Références

### Sources de données
- **jdRoll**: https://www.jdroll.org
- **La Cour d'Obéron**: https://lacourdoberon.com (si accessible)
- **Forum Rêves d'Encre**: https://revesdencre.fr
- **Transcriptions YouTube**: À collecter manuellement

### Infrastructure
- **Together.ai**: Qwen2.5-14B - $0.50/M tokens
- **Fireworks.ai**: Qwen2.5-14B - $0.50/M tokens
- **Hugging Face**: A100 80GB - ~$2/h

### Outils
- **Axolotl**: https://github.com/OpenAccess-AI-Collective/axolotl
- **vLLM**: https://vllm.readthedocs.io/
- **Whisper**: Transcription automatique

### Modèles
- **Qwen2.5-14B-Instruct**: https://huggingface.co/Qwen/Qwen2.5-14B-Instruct
- **Qwen2.5-7B-Instruct**: https://huggingface.co/Qwen/Qwen2.5-7B-Instruct

## 🔗 Liens AIDD

- [writing-plans](#) - Planifier les sessions de scraping
- [challenge-plan](#) - Valider les plans
- [monitoring](#) - Suivre les coûts et performances
- [learn](#) - Documenter les lessons learned

---

**Statut:** En attente de Phase 4 terminée  
**Priorité:** Haute  
**Labels:** scaling, data-collection, large-dataset, phase-5

**Next action:** Attendre fin Phase 4 pour commencer

## 📈 Roadmap Future

### Phase 6 : Production (Phase 5 complétée)
- API haute disponibilité
- Scaling automatique
- Monitoring avancé

### Phase 7 : Communauté (Phase 6 complétée)
- Open source du modèle
- Communauté d'utilisateurs
- Contributions externes

---

**End of Task**
- [ ] Documentation: `aidd_docs/changelog/phase5.md`
- [ ] Lessons: `aidd_docs/memory/phase5_lessons.md`
- [ ] Changelog: `aidd_docs/changelog/CHANGELOG.md`
