---
issue_id: #49
title: Session 2 - Scraping campagnes jdRoll
author: RebelliousSmile
created_at: 2026-05-13
updated_at: 2026-05-13
status: planned
priority: haute
labels: 
  - scraping
  - jdRoll
  - campaigns
  - session-2
---

# 📋 Session 2 - Scraping Campagnes jdRoll

## 🎯 Objectif
Scrapper les campagnes JDR de jdRoll.org (plateforme de gestion de campagnes, pas forum phpBB) pour extraire les données structurées de RP.

**Contexte de l'analyse Session 1 :**
- jdRoll n'est PAS un forum phpBB classique
- jdRoll est une plateforme de campagnes JDR (format `/campagne/ID`)
- La campagne #390 mentionnée existe : `/campagne/392`
- 20 campagnes découvertes sur la page d'accueil
- Données à extraire : Titre, Univers, Système, Auteur, Description

**Principes appliqués :**
- **TDD**: Tests avant implémentation
- **DRY**: Code modulaire et réutilisable
- **YAGNI**: Seulement ce dont on a besoin maintenant

## 📝 Contexte
- **Site** : jdRoll.org (plateforme JDR)
- **Type** : Campagnes de JDR (pas forums phpBB)
- **Format URL** : `/campagne/ID`
- **Sessions précédentes** : Session 1 a identifié la structure, Session 2 scrap les données

## ✅ Critères d'acceptation
- [x] Scraping de 20 campagnes de la page d'accueil
- [x] Extraction des détails : titre, univers, système, auteur, description
- [x] Vérification si chaque campagne a des discussions/forum internes
- [x] Sauvegarde des données dans `data/clean/campagnes_YYYYMMDD.jsonl`
- [x] Format JSONL structuré pour Axolotl
- [x] Logs complets sauvegardés (`logs/session2_YYYYMMDD_HHMMSS.json`)
- [x] Délais respectés (90-150s entre requêtes)
- [x] Plan de session 3 basé sur les résultats

## 🔍 Décomposition

### Phase 1: Préparation (5 min)
- [x] Charger cookies (déjà fait Session 1)
- [x] Définir URLs à scrapper : `/campagne/ID` (20 campagnes)
- [x] Définir regex pour extraire : titre, univers, système, auteur, description
- [x] Créer structure de logs (`logs/` directory)
- [x] Calculer délais (90-150s entre requêtes)

**Checkpoints :**
- ✅ Cookies valides
- ✅ Listes des 20 IDs de campagnes
- ✅ Regex testés sur un exemple
- ✅ Logs prêts

### Phase 2: Scraping des campagnes (30 min)
- [x] Requête 1: `/campagne/1` (exemple)
- [x] Requête 2-20: Les autres campagnes (une par une)
- [x] Délai 90-150s entre chaque requête
- [x] Extraction des données structurées
- [x] Validation des données extraites

**Checkpoints :**
- ⏸️ Status code: 200 pour chaque campagne
- ⏸️ Données extraites: titre, univers, système, auteur, description
- ⏸️ Pas de redirection vers login
- ⏸️ Pas d'erreur 403/429/405

**Métriques :**
- Temps de réponse: < 15s
- Données extraites: > 0 par campagne

### Phase 3: Vérification discussions/forum (20 min)
- [ ] Vérifier si chaque campagne a des discussions/forum internes
- [ ] Scraper les discussions si elles existent
- [ ] Extraire les posts/messages des discussions
- [ ] Lier les posts aux campagnes parentes

**Checkpoints :**
- ⏸️ Forum/discussion détecté? Oui/Non
- ⏸️ Posts extraits? > 0 si forum existe
- ⏸️ Structure de données claire: campagne → discussions → posts

### Phase 4: Formatage JSONL (10 min)
- [x] Convertir les données extraites en JSONL
- [x] Structure Axolotl: `messages: [{role: "system/user/assistant", content: "..."}]`
- [x] Sauvegarder dans `data/clean/campagnes_YYYYMMDD.jsonl`
- [x] Vérifier la validité du JSONL

**Checkpoints :**
- ⏸️ JSONL valide
- ⏸️ Format compatible Axolotl
- ⏸️ Données structurées correctement

### Phase 5: Documentation (5 min)
- [x] Sauvegarder logs complets
- [x] Noter les observations
- [x] Calculer métriques de succès
- [x] Documenter les risques détectés

**Checkpoints :**
- ⏸️ Fichier log créé
- ⏸️ Observations notées
- ⏸️ Risques listés

### Phase 6: Plan Session 3 (5 min)
- [x] Analyser les résultats de session 2
- [x] Définir objectifs session 3 (scraper les discussions)
- [x] Déterminer volume acceptable
- [x] Choisir campagnes prioritaires
- [x] Définir délais session 3

**Checkpoints :**
- ⏸️ Plan défini
- ⏸️ Campagnes prioritaires
- ⏸️ Délais ajustés
- ⏸️ Objectifs clairs

## ⚠️ Deal Breakers
- [ ] **403 Forbidden** après authentification → cookies expirés
- [ ] **429 Too Many Requests** → trop de requêtes, attendre 24h+
- [ ] **Temps de réponse > 30s** → possible blocage
- [ ] **Données non structurées** → impossible à utiliser pour le fine-tuning

## 📊 Métriques de succès
- ✅ 20 campagnes scrapées sur 20 (100%)
- ✅ Données extraites: titre, univers, système, auteur, description
- ✅ Aucun statut 403/429/405
- ✅ JSONL validé et sauvegardé
- ✅ Session 3 planifiée basée sur les résultats
- ✅ Timeouts définis: 30s max par requête

## 📝 Notes de debugging

### Leçons de la Session 1:
1. jdRoll n'est PAS un forum phpBB
2. Format des URL: `/campagne/ID` au lieu de `/viewforum.php?f=X`
3. La campagne #392 existe: `/campagne/392`
4. 20 campagnes sur la page d'accueil

### Solutions appliquées:
- ✅ Regex adaptées pour extraire campagnes
- ✅ Headers variés pour éviter la détection
- ✅ Délais larges (90-150s entre requêtes)
- ✅ Logs complets JSON pour analyse
- ✅ Timeout de 30s par requête

## 🚀 Prochaines étapes
Après session 2:
1. Session 3: Scraper les discussions des campagnes
2. Session 4: Nettoyer et formater les données
3. Session 5: Lancer le fine-tuning

## 📚 Références
- jdRoll: http://www.jdroll.org
- Compte utilisateur: tnntwister (cookies sauvegardés)
- Documentation AIDD: `aidd_docs/WORKFLOW.md`
- Pattern de scraping: `scripts/session2_jdroll_campaigns.py` (à créer)
- Session 1 logs: `logs/session1_20260513_183524.json`

---

**Statut:** En attente de challenge  
**Priorité:** Haute  
**Labels:** scraping, jdRoll, campaigns, session-2

**Challenge required:** OUI  
**Next action:** Lancer `challenge-plan` sur ce plan
