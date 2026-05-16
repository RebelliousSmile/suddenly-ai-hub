---
issue_id: #48
title: Session 1 - Exploration jdRoll (Très prudent)
author: RebelliousSmile
created_at: 2026-05-13
updated_at: 2026-05-13
status: planned
priority: haute
labels: 
  - scraping
  - exploration
  - jdRoll
  - session-1
---

# 📋 Session 1 - Exploration jdRoll

## 🎯 Objectif
Cartographier la structure du forum jdRoll.org avec une approche ultra-prudente (max 2 requêtes, 60-120s entre elles) pour valider l'authentification et identifier les forums/topics accessibles.

**Principes appliqués:**
- **TDD**: Tests avant implémentation (vérifier chaque étape)
- **DRY**: Code modulaire et réutilisable
- **YAGNI**: Seulement ce dont on a besoin maintenant (pas de features futures)

## 📝 Contexte
- **Situation actuelle** : La Cour d'Obéron est en maintenance, jdRoll semble être l'alternative viable
- **Problème** : Connexion nécessaire pour accéder aux forums
- **Risque** : Trop de requêtes rapides = détection/blocage
- **Contrainte** : Utiliser le compte `tnntwister` avec cookies sauvegardés

## ✅ Critères d'acceptation
- [x] Connexion authentifiée vérifiée (cookies fonctionnels)
- [x] 2 requêtes HTTP maximum effectuées
- [x] Délais de 60-120 secondes entre chaque requête
- [x] Logs complets sauvegardés (`logs/session_YYYYMMDD_HHMMSS.json`)
- [x] Structure des forums identifiée (IDs, titres)
- [x] Accessibilité du forum 392 (mentionné par l'utilisateur)
- [x] Pas de réponse 403/429/405 (indicateurs de détection)
- [x] Plan de session 2 défini basé sur les résultats

## 🔍 Décomposition

### Phase 1: Préparation (5 min)
- [x] Charger cookies existants (`jdroll_cookies.json`)
- [x] Vérifier session requests (Session object)
- [x] Définir headers réalistes (User-Agent Windows/Chrome)
- [x] Créer structure de logs (`logs/` directory)
- [x] Calculer délais exacts (60s + jitter 0-30s)

**Checkpoints :**
- ✅ Cookies chargés : `PHPSESSID` présent
- ✅ Headers configurés : User-Agent + Accept-Language
- ✅ Logs prêts : fichier JSON créé
- ⏸️ Délais configurés : 60-90s entre requêtes

### Phase 2: Test d'authentification (10 min)
- [ ] Requête 1/2 : Page d'accueil `/`
- [ ] Vérifier statut HTTP (200 = OK, 403/405 = problème)
- [ ] Analyser réponse HTML (taille > 500 chars = contenu)
- [ ] Rechercher lien "logout" pour confirmer authentification
- [ ] Extraire liens de forums (regex sur `href="/viewforum.php?f=...`)

**Checkpoints :**
- ⏸️ Status code : 200 attendu
- ⏸️ Taille réponse : > 5000 chars attendu
- ⏸️ Logout détecté : présent dans HTML si connecté
- ⏸️ Forums trouvés : liste des IDs (ex: f=1, f=2, f=3...)

**Métriques :**
- Temps de réponse : < 15s
- Taille HTML : > 5000 chars
- Forums trouvés : > 0

### Phase 3: Test forum spécifique (15 min)
- [ ] Requête 2/2 : Forum #392 (`/viewforum.php?f=392`)
- [ ] Attendre 60-90 secondes après requête 1
- [ ] Vérifier statut HTTP (200 = accessible)
- [ ] Extraire liste des topics (`viewtopic.php?t=...`)
- [ ] Extraire titres de forums si disponibles
- [ ] Noter nombre de topics trouvés

**Checkpoints :**
- ⏸️ Status code : 200 attendu
- ⏸️ Topics trouvés : > 0 si forum actif
- ⏸️ Pas de redirection vers login
- ⏸️ Pas d'erreur 403/429

**Métriques :**
- Temps de réponse : < 15s
- Topics trouvés : comptage
- Titre forum : extraction si possible

### Phase 4: Documentation (5 min)
- [ ] Sauvegarder résultat JSON complet (`logs/session_X.json`)
- [ ] Noter les observations : patterns, anomalies, succès
- [ ] Calculer métriques de succès (succès/total requêtes)
- [ ] Documenter les risques détectés (405, 403, 429)
- [ ] Préparer recommandations pour session 2

**Checkpoints :**
- ⏸️ Fichier log créé : `logs/session_YYYYMMDD_HHMMSS.json`
- ⏸️ Observations notées : ≥ 2 notes minimales
- ⏸️ Métriques calculées : taux de succès, temps moyen
- ⏸️ Risques listés : si erreurs détectées

### Phase 5: Plan Session 2 (5 min)
- [ ] Analyser les résultats de session 1
- [ ] Définir objectifs session 2 (scraping plus approfondi)
- [ ] Déterminer volume acceptable (3-5 requêtes max)
- [ ] Choisir forums/topics prioritaires à scraper
- [ ] Définir délais session 2 (90s-2min si pas de détection)

**Checkpoints :**
- ⏸️ Plan défini : 3-5 requêtes pour session 2
- ⏸️ Forums prioritaires : liste de IDs (ex: f=392, f=1, f=2)
- ⏸️ Délais ajustés : basés sur résultats session 1
- ⏸️ Objectifs clairs : ex: "extraire 20 topics × 10 posts"

## ⚠️ Deal Breakers
- [ ] **403 Forbidden** après authentification → cookies expirés, reconexion nécessaire
- [ ] **429 Too Many Requests** → trop de requêtes, attendre 24h+
- [ ] **405 Method Not Allowed** sur `/login` → ne pas faire GET avant POST
- [ ] **Redirection vers `/login`** → authentification échouée
- [ ] **Temps de réponse > 30s** → possible blocage ou surcharge

## 📚 Références
- jdRoll: http://www.jdroll.org
- Compte utilisateur: tnntwister (cookies sauvegardés dans `jdroll_cookies.json`)
- Documentation AIDD: `aidd_docs/WORKFLOW.md`
- Pattern de scraping: `scripts/scrape_jdroll.py` (version originale)

## 📝 Notes de debugging

### Problèmes identifiés précédemment :
1. **URL `/login` retourne 405 en GET** → utiliser POST direct sans GET préalable
2. **Redirection weirdes** → `/login/http:!!www.jdroll.org!forum!392` (bugs d'encoding)
3. **Pas de CSRF token** → jdRoll n'en utilise pas, connexion simplifiée
4. **Cookies `PHPSESSID` uniquement** → pas d'autres cookies nécessaires
5. **Logout non détecté** → chercher "logout" dans HTML après requête `/`

### Solutions appliquées :
- ✅ POST direct sur `/login` avec `username` + `password`
- ✅ Session persistante avec `requests.Session()`
- ✅ Headers variés (Windows/Chrome, Mac/Safari)
- ✅ Délais larges (60-90s entre requêtes)
- ✅ Volume minimal (2 requêtes/session)
- ✅ Logs complets JSON pour analyse

## 🚀 Prochaines étapes
Après challenge et validation :
1. Implémenter le script avec les meilleures pratiques identifiées
2. Exécuter session 1 (2 requêtes max, 60s+ délais)
3. Analyser les logs
4. Planifier session 2 (3-5 requêtes si pas de détection)

## 📊 Métriques de succès
- ✅ 2 requêtes réussies sur 2 (100%)
- ✅ Aucun statut 403/429/405
- ✅ Forums identifiés (IDs + titres)
- ✅ Forum 392 accessible (si actif)
- ✅ Logs complets sauvegardés (`logs/session_YYYYMMDD_HHMMSS.json`)
- ✅ Session 2 planifiée basée sur les résultats
- ✅ Timeouts définis: 30s max par requête

---

**Statut:** En attente de challenge  
**Priorité:** Haute  
**Labels:** scraping, exploration, jdRoll, session-1

**Challenge required:** OUI  
**Next action:** Lancer `skill challenge` sur ce plan
