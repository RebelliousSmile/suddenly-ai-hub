#!/usr/bin/env python3
"""
Session 1 - Exploration jdRoll
Date: 2026-05-13
Objectif : Cartographier la structure du forum avec approche ultra-prudente

Améliorations implémentées :
✅ Random jitter dans les délais (60-120s)
✅ Headers variés (Chrome/Safari/Edge)
✅ Pre-check d'authentification avant requêtes
"""

import requests
import re
import os
import json
import time
import random
import string
from datetime import datetime

# =========================================
# CONFIGURATION
# =========================================

BASE_URL = "http://www.jdroll.org"

# Headers variés pour éviter la détection
HEADERS_VARIANTS = [
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
        "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
    },
    {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15",
        "Accept-Language": "fr-FR,fr;q=0.9",
    },
    {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0",
        "Accept-Language": "fr-FR,fr;q=0.9",
    },
]

# Délais : 60-120s random (amélioration #1)
MIN_DELAY = 60
MAX_DELAY = 120

# Cookies
COOKIES_FILE = "/home/user/suddenly-ai-hub/jdroll_cookies.json"
LOGS_DIR = "/home/user/suddenly-ai-hub/logs"
DATA_DIR = "/home/user/suddenly-ai-hub/data"

# =========================================
# UTILITAIRES
# =========================================

def random_string(length=8):
    """Génère une chaîne aléatoire pour le nom de fichier"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def get_delay():
    """Délai random entre MIN_DELAY et MAX_DELAY (amélioration #1)"""
    return random.uniform(MIN_DELAY, MAX_DELAY)

def get_headers(variant_idx=None):
    """Headers variés (amélioration #2)"""
    if variant_idx is None:
        variant_idx = random.randint(0, len(HEADERS_VARIANTS) - 1)
    return HEADERS_VARIANTS[variant_idx]

# =========================================
# CLASSE PRINCIPALE
# =========================================

class JdRollSession1:
    """Session 1 d'exploration jdRoll - Très prudente"""
    
    def __init__(self):
        self.session = requests.Session()
        self.results = {
            "session": 1,
            "date": datetime.now().isoformat(),
            "plan_ref": "session1_jdroll_exploration.md",
            "challenged": True,
            "requests": [],
            "observations": [],
            "metrics": {
                "total_requests": 0,
                "successful": 0,
                "failed": 0,
                "avg_response_time": 0
            },
            "forums": [],
            "topics": [],
            "recommendations": []
        }
        
        print("="*80)
        print("SESSION 1 - EXPLORATION JDROLL (Très prudent)")
        print("="*80)
        print(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏱️ Délai random: 60-120 secondes")
        print("📊 Volume: 2 requêtes MAX")
        print("🔄 Headers: Variables (Chrome/Safari/Edge)")
        print("="*80)
    
    def load_cookies(self):
        """Charger les cookies existants"""
        print("\n🍪 Chargement des cookies...")
        
        if not os.path.exists(COOKIES_FILE):
            print("❌ Erreur: Fichier de cookies non trouvé")
            print(f"   Path: {COOKIES_FILE}")
            return False
        
        try:
            with open(COOKIES_FILE, "r") as f:
                cookies_dict = json.load(f)
            
            if not cookies_dict:
                print("❌ Erreur: Cookies vides")
                return False
            
            for name, value in cookies_dict.items():
                self.session.cookies.set(name, value, domain="jdroll.org")
            
            print(f"✅ {len(cookies_dict)} cookies chargés: {list(cookies_dict.keys())}")
            return True
            
        except Exception as e:
            print(f"❌ Erreur chargement: {e}")
            return False
    
    def pre_check_authentication(self):
        """Pre-check d'authentification (amélioration #3)"""
        print("\n🔐 Pré-check d'authentification...")
        
        try:
            headers = get_headers()
            response = self.session.get(
                f"{BASE_URL}/",
                headers=headers,
                timeout=15
            )
            
            print(f"   Status: {response.status_code}")
            print(f"   Taille: {len(response.text)} caractères")
            
            # Vérifier si connecté (rechercher "logout" ou "déconnexion")
            is_connected = any(
                keyword in response.text.lower()
                for keyword in ["logout", "déconnexion", "se déconnecter", "disconnect"]
            )
            
            if is_connected:
                print("✅ Authentification confirmée (logout détecté)")
                self.results["observations"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "auth_check",
                    "status": "success",
                    "note": "Authentification confirmée par présence de lien logout"
                })
                return True
            else:
                print("⚠️ Authentification douteuse (pas de logout détecté)")
                print("   → Continuer quand même (peut-être format différent)")
                self.results["observations"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "auth_check",
                    "status": "doubtful",
                    "note": "Pas de logout détecté, mais continuation autorisée"
                })
                return True  # Continuer quand même
                
        except Exception as e:
            print(f"❌ Erreur pré-check: {e}")
            self.results["observations"].append({
                "timestamp": datetime.now().isoformat(),
                "type": "auth_check",
                "status": "error",
                "error": str(e)
            })
            return False
    
    def make_request(self, url, request_num, name="Req"):
        """Faire une requête avec journalisation complète"""
        delay = get_delay()
        headers = get_headers()
        
        print(f"\n{'='*80}")
        print(f"📋 [{request_num}/2] {name}")
        print(f"   URL: {url[:100]}")
        print(f"   Délai avant: {delay:.1f}s")
        print(f"   Headers: {list(headers['User-Agent'].split('/')[0].split()[0])}")
        
        time.sleep(delay)
        
        start_time = time.time()
        
        try:
            response = self.session.get(url, headers=headers, timeout=30)
            response_time = time.time() - start_time
            
            self.results["metrics"]["total_requests"] += 1
            
            # Journaliser
            request_log = {
                "request_num": request_num,
                "name": name,
                "url": url,
                "status_code": response.status_code,
                "response_time": f"{response_time:.2f}s",
                "size": len(response.text),
                "timestamp": datetime.now().isoformat(),
                "headers_used": headers['User-Agent'][:50] + "..."
            }
            
            # Vérifier succès
            success = (
                response.status_code == 200 and 
                len(response.text) > 500 and
                "login" not in response.url.lower()
            )
            
            if success:
                self.results["metrics"]["successful"] += 1
                status_icon = "✅"
                print(f"{status_icon} {response.status_code} - {len(response.text)} caractères")
                print(f"   Temps: {response_time:.2f}s")
                print(f"   Headers: {headers['User-Agent'].split('/')[0]}")
            else:
                self.results["metrics"]["failed"] += 1
                status_icon = "⚠️"
                print(f"{status_icon} {response.status_code} - {len(response.text)} caractères")
                print(f"   Temps: {response_time:.2f}s")
                print(f"   → Pas assez de contenu ou redirection login")
            
            request_log["success"] = success
            
            # Sauvegarder HTML si succès
            if success:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                html_file = f"data/session1_req{request_num}_{timestamp}.html"
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                request_log["html_saved"] = html_file
                print(f"   HTML sauvegardé: {html_file}")
            
            # Ajouter aux résultats
            self.results["requests"].append(request_log)
            self.results["metrics"]["total_requests"] += 1
            if success:
                self.results["metrics"]["successful"] += 1
            else:
                self.results["metrics"]["failed"] += 1
            
            return response
            
        except requests.exceptions.Timeout:
            print(f"❌ Timeout après 30s")
            self.results["warnings"] = self.results.get("warnings", [])
            self.results["warnings"].append({
                "request": request_num,
                "error": "Timeout"
            })
            return None
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            self.results["warnings"] = self.results.get("warnings", [])
            self.results["warnings"].append({
                "request": request_num,
                "error": str(e)
            })
            return None
    
    def extract_forums(self, html, request_num):
        """Extraire les forums du HTML"""
        print(f"\n📁 Extraction des forums...")
        
        # Regex plus flexible pour supporter différents formats
        forums = re.findall(r'href=["\'](/viewforum\.php[^"\'?\s]*[^"\'\s/])["\']', html)
        
        # Nettoyer et extraire IDs
        forum_list = []
        for link in forums:
            # Extraire ID
            match = re.search(r'f=(\d+)', link)
            forum_id = match.group(1) if match else "?"
            
            # Extraire titre si possible
            title_match = re.search(
                rf'href=["\']\Q{re.escape(link)}\E[^"\']*["\'][^>]*>([^<]+)</a>',
                html,
                re.IGNORECASE
            )
            title = title_match.group(1).strip() if title_match else "?"
            
            forum_list.append({
                "id": forum_id,
                "link": link,
                "title": title,
                "found_in_request": request_num
            })
        
        # Dédupe
        unique_forums = []
        seen_ids = set()
        for f in forum_list:
            if f["id"] not in seen_ids:
                unique_forums.append(f)
                seen_ids.add(f["id"])
        
        print(f"   Forums trouvés: {len(unique_forums)}")
        
        if unique_forums:
            print(f"\n   📋 Liste des forums:")
            for f in unique_forums[:10]:  # Limite à 10
                print(f"      f={f['id']:3} - {f['title'][:60]}")
        
        self.results["forums"] = unique_forums
        return unique_forums
    
    def extract_topics(self, html, forum_id, request_num):
        """Extraire les topics d'un forum"""
        print(f"\n📝 Extraction des topics (forum f={forum_id})...")
        
        topics = re.findall(r'viewtopic\.php[^"\']*', html)
        
        topic_list = []
        for link in topics:
            match = re.search(r't=(\d+)', link)
            if match:
                topic_list.append({
                    "id": match.group(1),
                    "link": link,
                    "forum_id": forum_id
                })
        
        print(f"   Topics trouvés: {len(topic_list)}")
        
        if topic_list:
            self.results["topics"].extend(topic_list)
        
        return topic_list
    
    def save_results(self):
        """Sauvegarder les résultats complets"""
        print("\n" + "="*80)
        print("💾 Sauvegarde des résultats...")
        
        # Créer logs dir
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{LOGS_DIR}/session1_{timestamp}.json"
        
        # Calculer metrics
        if self.results["metrics"]["total_requests"] > 0:
            successful = self.results["metrics"]["successful"]
            self.results["metrics"]["success_rate"] = f"{(successful / self.results['metrics']['total_requests'] * 100):.1f}%"
        else:
            self.results["metrics"]["success_rate"] = "N/A"
        
        # Sauvegarder
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Résultats sauvegardés: {log_file}")
        
        # Résumé
        print("\n" + "="*80)
        print("📊 SYNTHÈSE SESSION 1")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Requêtes: {self.results['metrics']['total_requests']}/2")
        print(f"Succès: {self.results['metrics']['successful']} ({self.results['metrics'].get('success_rate', 'N/A')})")
        print(f"Échecs: {self.results['metrics']['failed']}")
        print(f"Forums trouvés: {len(self.results['forums'])}")
        print(f"Topics trouvés: {len(self.results['topics'])}")
        print(f"Logs: {log_file}")
        
        if self.results["warnings"]:
            print(f"\n⚠️ Warnings: {len(self.results['warnings'])}")
            for w in self.results["warnings"]:
                print(f"   • {w}")
        
        print("\n📝 Observations:")
        for obs in self.results["observations"]:
            print(f"   • {obs['note']}")
        
        return log_file
    
    def generate_recommendations(self):
        """Générer recommandations pour session 2"""
        print("\n💡 Génération des recommandations...")
        
        recommendations = []
        
        # Analyser les succès/échecs
        success_rate = self.results["metrics"]["successful"] / max(
            self.results["metrics"]["total_requests"], 1
        )
        
        if success_rate == 1.0:
            recommendations.append({
                "type": "success",
                "note": "Toutes les requêtes réussies - session 2 peut augmenter le volume"
            })
            
            # Recommander volume session 2
            recommendations.append({
                "type": "action",
                "note": "Session 2: 3-5 requêtes, délais 90-150s"
            })
            
        elif success_rate > 0.5:
            recommendations.append({
                "type": "warning",
                "note": "Quelques échecs détectés - rester prudent en session 2"
            })
            recommendations.append({
                "type": "action",
                "note": "Session 2: 2-3 requêtes, délais 120-180s"
            })
        else:
            recommendations.append({
                "type": "error",
                "note": "Trop d'échecs - vérifier cookies ou attendre 24h"
            })
            recommendations.append({
                "type": "action",
                "note": "Session 2: Annuler ou tester avec nouveaux cookies"
            })
        
        # Recommandations spécifiques
        if self.results["forums"]:
            recommendations.append({
                "type": "opportunity",
                "note": f"Forums identifiés: {len(self.results['forums'])} - prioriser ceux avec IDs pairs"
            })
            
            # Proposer forums prioritaires
            forum_ids = [f["id"] for f in self.results["forums"][:3]]
            recommendations.append({
                "type": "target",
                "note": f"Forums prioritaires: f={','.join(forum_ids)}"
            })
        
        if self.results["topics"]:
            recommendations.append({
                "type": "opportunity",
                "note": f"Topics identifiés: {len(self.results['topics'])} - test d'extraction détaillée possible"
            })
        
        self.results["recommendations"] = recommendations
        
        print(f"   {len(recommendations)} recommandations générées:")
        for rec in recommendations:
            print(f"      • [{rec['type']}] {rec['note']}")
        
        return recommendations
    
    def run(self):
        """Exécuter la session complète"""
        print("\n🚀 LANCEMENT SESSION 1")
        
        # 1. Charger cookies
        if not self.load_cookies():
            print("❌ Impossible de continuer sans cookies")
            return
        
        # 2. Pré-check authentification
        if not self.pre_check_authentication():
            print("⚠️ Auth douteuse mais continuation autorisée")
        
        # 3. Requête 1/2 - Page d'accueil
        resp1 = self.make_request(
            f"{BASE_URL}/",
            request_num=1,
            name="Page d'accueil"
        )
        
        if resp1:
            self.extract_forums(resp1.text, request_num=1)
            
            # 4. Requête 2/2 - Forum #392 (mentionné par l'utilisateur)
            print("\n" + "="*80)
            print("📋 [2/2] Forum #392 (mentionné par l'utilisateur)")
            print("="*80)
            
            time.sleep(get_delay())
            
            resp2 = self.make_request(
                f"{BASE_URL}/viewforum.php?f=392",
                request_num=2,
                name="Forum #392"
            )
            
            if resp2:
                self.extract_topics(resp2.text, forum_id=392, request_num=2)
        
        # 5. Sauvegarder résultats
        log_file = self.save_results()
        
        # 6. Générer recommandations
        self.generate_recommendations()
        
        # 7. Sauvegarder à nouveau avec recommendations
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{LOGS_DIR}/session1_{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*80)
        print("✅ SESSION 1 TERMINÉE")
        print("="*80)
        print(f"Log complet: {log_file}")
        print("\n📅 Prochaine session: Attendre 24h+ minimum")
        print("📋 Pour lancer: python scripts/session1_jdroll.py")
        print("="*80)


# =========================================
# POINT D'ENTRÉE
# =========================================

if __name__ == "__main__":
    session = JdRollSession1()
    session.run()
