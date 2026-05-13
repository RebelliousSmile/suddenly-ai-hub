#!/usr/bin/env python3
"""
Session 2 - Scraping Campagnes jdRoll
Date: 2026-05-13
Objectif : Scrapper les 20 campagnes JDR de jdRoll.org

Améliorations implémentées :
✅ Random jitter dans les délais (90-150s)
✅ Headers variés (Chrome/Safari/Edge)
✅ Adapté à la structure jdRoll (campagnes au lieu de forums)
✅ Extraction: titre, univers, système, auteur, description
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

# Délais : 90-150s random
MIN_DELAY = 90
MAX_DELAY = 150

# Cookies
COOKIES_FILE = "/home/user/suddenly-ai-hub/jdroll_cookies.json"
LOGS_DIR = "/home/user/suddenly-ai-hub/logs"
DATA_DIR = "/home/user/suddenly-ai-hub/data/clean"

# =========================================
# UTILITAIRES
# =========================================

def random_string(length=8):
    """Génère une chaîne aléatoire pour le nom de fichier"""
    return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

def get_delay():
    """Délai random entre MIN_DELAY et MAX_DELAY"""
    return random.uniform(MIN_DELAY, MAX_DELAY)

def get_headers(variant_idx=None):
    """Headers variés"""
    if variant_idx is None:
        variant_idx = random.randint(0, len(HEADERS_VARIANTS) - 1)
    return HEADERS_VARIANTS[variant_idx]

# =========================================
# CLASSE PRINCIPALE
# =========================================

class JdRollSession2:
    """Session 2 - Scraping des campagnes JDR"""
    
    def __init__(self):
        self.session = requests.Session()
        self.results = {
            "session": 2,
            "date": datetime.now().isoformat(),
            "plan_ref": "session2_jdroll_campaigns.md",
            "challenged": True,
            "requests": [],
            "observations": [],
            "campaigns": [],
            "metrics": {
                "total_requests": 0,
                "successful": 0,
                "failed": 0,
                "success_rate": "N/A"
            },
            "warnings": []
        }
        
        print("="*80)
        print("SESSION 2 - SCRAPING CAMPAGNES JDROLL")
        print("="*80)
        print(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏱️ Délai random: 90-150 secondes")
        print("📊 Volume: 20 campagnes MAX")
        print("🔄 Headers: Variables (Chrome/Safari/Edge)")
        print("="*80)
    
    def load_cookies(self):
        """Charger les cookies existants"""
        print("\n🍪 Chargement des cookies...")
        
        if not os.path.exists(COOKIES_FILE):
            print("❌ Erreur: Fichier de cookies non trouvé")
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
        """Vérifier l'authentification"""
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
            
            # Vérifier si connecté
            is_connected = any(
                keyword in response.text.lower()
                for keyword in ["logout", "déconnexion", "disconnect", "tnntwister"]
            )
            
            if is_connected:
                print("✅ Authentification confirmée")
                self.results["observations"].append({
                    "timestamp": datetime.now().isoformat(),
                    "type": "auth_check",
                    "status": "success",
                    "note": "Authentification confirmée"
                })
                return True
            else:
                print("⚠️ Authentification douteuse")
                return True  # Continuer quand même
                
        except Exception as e:
            print(f"❌ Erreur pré-check: {e}")
            return False
    
    def extract_campaigns_from_homepage(self, html):
        """Extraire la liste des 20 campagnes de la page d'accueil"""
        print("\n📁 Extraction des campagnes de la page d'accueil...")
        
        # Regex adapté pour jdRoll
        campaign_links = re.findall(r'href=["\'](/campagne/(\d+))["\']', html)
        
        campaigns = []
        for link, id_ in campaign_links:
            # Extraire le titre et la description
            title_match = re.search(
                rf'href=["\']\Q{link}\E[^"\']*["\'][^>]*>([^<]+)</a>',
                html,
                re.IGNORECASE
            )
            title = title_match.group(1).strip() if title_match else "?"
            
            # Extraire la description
            desc_match = re.search(
                rf'href=["\']\Q{link}\E[^"\']*["\'].*?<p>(.*?)</p>',
                html,
                re.IGNORECASE | re.DOTALL
            )
            desc = desc_match.group(1).strip() if desc_match else "?"
            
            # Nettoyer le HTML
            desc = re.sub(r'<[^>]+>', ' ', desc)
            desc = ' '.join(desc.split())[:200]  # Limiter à 200 caractères
            
            # Extraire l'auteur
            author_match = re.search(
                rf'href=["\']\Q{link}\E[^"\']*["\'].*?<i>Proposé par (.*?)</i>',
                html,
                re.IGNORECASE | re.DOTALL
            )
            author = author_match.group(1).strip() if author_match else "Inconnu"
            
            # Extraire l'univers et le système (simplifié)
            universe_system = desc.split('avec le système')
            if len(universe_system) > 1:
                universe = universe_system[0].replace('Dans l\'univers de ', '').strip()
                system = universe_system[1].split('.')[0].strip() + "."
            else:
                universe = "Inconnu"
                system = "Inconnu"
            
            campaigns.append({
                "id": id_,
                "link": link,
                "title": title,
                "universe": universe,
                "system": system,
                "author": author,
                "description": desc,
                "found_in_request": "homepage"
            })
        
        print(f"✅ Campagnes trouvées: {len(campaigns)}")
        return campaigns
    
    def make_request(self, url, request_num, name="Req"):
        """Faire une requête avec journalisation complète"""
        delay = get_delay()
        headers = get_headers()
        
        print(f"\n{'='*80}")
        print(f"📋 [{request_num}/20] {name}")
        print(f"   URL: {url[:100]}")
        print(f"   Délai avant: {delay:.1f}s")
        print(f"   Headers: {headers['User-Agent'][:50]}...")
        
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
            
            request_log["success"] = success
            self.results["requests"].append(request_log)
            
            return response
            
        except requests.exceptions.Timeout:
            print(f"❌ Timeout après 30s")
            self.results["warnings"].append({
                "request": request_num,
                "error": "Timeout"
            })
            return None
            
        except Exception as e:
            print(f"❌ Erreur: {e}")
            self.results["warnings"].append({
                "request": request_num,
                "error": str(e)
            })
            return None
    
    def scrape_campaign_details(self, campaign, request_num):
        """Scrapper les détails complets d'une campagne"""
        try:
            response = self.make_request(
                f"{BASE_URL}{campaign['link']}",
                request_num=request_num,
                name=f"Campagne #{campaign['id']}"
            )
            
            if response and response.status_code == 200:
                # Extraire plus de détails si possible
                html = response.text
                
                # Vérifier si la campagne a des discussions/forum internes
                has_discussions = "discussion" in html.lower() or "forum" in html.lower() or "posts" in html.lower()
                
                campaign_data = {
                    "id": campaign["id"],
                    "title": campaign["title"],
                    "universe": campaign["universe"],
                    "system": campaign["system"],
                    "author": campaign["author"],
                    "description": campaign["description"],
                    "link": campaign["link"],
                    "has_discussions": has_discussions,
                    "html_size": len(html),
                    "scraped_at": datetime.now().isoformat()
                }
                
                # Sauvegarder HTML
                html_file = f"data/clean/campaign_{campaign['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                os.makedirs(os.path.dirname(html_file), exist_ok=True)
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(html)
                
                campaign_data["html_saved"] = html_file
                
                return campaign_data
                
        except Exception as e:
            self.results["warnings"].append({
                "campaign_id": campaign["id"],
                "error": str(e)
            })
            return None
        
        return None
    
    def save_results(self):
        """Sauvegarder les résultats complets"""
        print("\n" + "="*80)
        print("💾 Sauvegarde des résultats...")
        
        # Créer directories
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        
        # Calculer metrics
        total = self.results["metrics"]["total_requests"]
        if total > 0:
            successful = self.results["metrics"]["successful"]
            self.results["metrics"]["success_rate"] = f"{(successful / total * 100):.1f}%"
        else:
            self.results["metrics"]["success_rate"] = "N/A"
        
        # Nom de fichier
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{LOGS_DIR}/session2_{timestamp}.json"
        data_file = f"{DATA_DIR}/campagnes_{timestamp}.json"
        
        # Sauvegarder
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(self.results["campaigns"], f, indent=2, ensure_ascii=False)
        
        print(f"✅ Résultats sauvegardés:")
        print(f"   • Logs: {log_file}")
        print(f"   • Données: {data_file}")
        
        # Résumé
        print("\n" + "="*80)
        print("📊 SYNTHÈSE SESSION 2")
        print("="*80)
        print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"Requêtes: {self.results['metrics']['total_requests']}/20")
        print(f"Succès: {self.results['metrics']['successful']} ({self.results['metrics'].get('success_rate', 'N/A')})")
        print(f"Échecs: {self.results['metrics']['failed']}")
        print(f"Campagnes scrapées: {len(self.results['campaigns'])}")
        
        if self.results["warnings"]:
            print(f"\n⚠️ Warnings: {len(self.results['warnings'])}")
            for w in self.results["warnings"]:
                print(f"   • {w}")
        
        print("\n📝 Observations:")
        for obs in self.results["observations"]:
            print(f"   • {obs['note']}")
        
        return log_file, data_file
    
    def generate_recommendations(self):
        """Générer recommandations pour session 3"""
        print("\n💡 Génération des recommandations...")
        
        recommendations = []
        success_rate = self.results["metrics"]["successful"] / max(
            self.results["metrics"]["total_requests"], 1
        )
        
        if success_rate >= 0.8:
            recommendations.append({
                "type": "success",
                "note": "Succès élevé - Session 3 peut scraper les discussions"
            })
            recommendations.append({
                "type": "action",
                "note": "Session 3: Scraper les discussions des campagnes (90-150s délais)"
            })
        else:
            recommendations.append({
                "type": "warning",
                "note": "Taux de succès faible - Rester prudent"
            })
            recommendations.append({
                "type": "action",
                "note": "Session 3: Augmenter les délais (120-180s)"
            })
        
        # Recommandations spécifiques
        if self.results["campaigns"]:
            campaigns_with_discussions = [
                c for c in self.results["campaigns"] 
                if c.get("has_discussions", False)
            ]
            
            if campaigns_with_discussions:
                recommendations.append({
                    "type": "opportunity",
                    "note": f"{len(campaigns_with_discussions)} campagnes ont des discussions - Scraper en Session 3"
                })
                
                # List les campagnes prioritaires
                discussion_ids = [c["id"] for c in campaigns_with_discussions[:3]]
                recommendations.append({
                    "type": "target",
                    "note": f"Campagnes avec discussions: {discussion_ids}"
                })
        
        self.results["recommendations"] = recommendations
        
        print(f"   {len(recommendations)} recommandations générées:")
        for rec in recommendations:
            print(f"      • [{rec['type']}] {rec['note']}")
        
        return recommendations
    
    def run(self):
        """Exécuter la session complète"""
        print("\n🚀 LANCEMENT SESSION 2")
        
        # 1. Charger cookies
        if not self.load_cookies():
            print("❌ Impossible de continuer sans cookies")
            return
        
        # 2. Pré-check authentification
        if not self.pre_check_authentication():
            print("⚠️ Auth douteuse mais continuation autorisée")
        
        # 3. Extraire la liste des campagnes
        print("\n" + "="*80)
        print("📋 EXTRACTION DE LA PAGE D'ACCUEIL")
        print("="*80)
        
        time.sleep(get_delay())
        
        homepage_resp = self.make_request(
            f"{BASE_URL}/",
            request_num=1,
            name="Page d'accueil"
        )
        
        if not homepage_resp or homepage_resp.status_code != 200:
            print("❌ Impossible de scraper la page d'accueil")
            return
        
        # Extraire les campagnes
        self.results["campaigns"] = self.extract_campaigns_from_homepage(homepage_resp.text)
        
        if not self.results["campaigns"]:
            print("❌ Aucune campagne trouvée")
            return
        
        print(f"\n✅ {len(self.results['campaigns'])} campagnes extraites")
        for camp in self.results["campaigns"][:5]:
            print(f"   • f={camp['id']}: {camp['title'][:60]}")
        
        # 4. Scraper les détails de chaque campagne
        print("\n" + "="*80)
        print("📋 SCRAPPING DES DÉTAILS DES CAMPAGNES")
        print("="*80)
        
        for i, campaign in enumerate(self.results["campaigns"], 1):
            if i > 1:  # Première requête déjà faite
                time.sleep(get_delay())
            
            campaign_details = self.scrape_campaign_details(campaign, request_num=i+1)
            
            if campaign_details:
                # Mettre à jour les données
                self.results["campaigns"][i-1].update(campaign_details)
        
        # 5. Sauvegarder résultats
        log_file, data_file = self.save_results()
        
        # 6. Générer recommandations
        self.generate_recommendations()
        
        # 7. Mettre à jour et sauvegarder
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{LOGS_DIR}/session2_{timestamp}.json"
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        print("\n" + "="*80)
        print("✅ SESSION 2 TERMINÉE")
        print("="*80)
        print(f"Log complet: {log_file}")
        print(f"Données: {data_file}")
        print("\n📅 Prochaine session: Session 3 - Scraper les discussions")
        print("="*80)


# =========================================
# POINT D'ENTRÉE
# =========================================

if __name__ == "__main__":
    session = JdRollSession2()
    session.run()
