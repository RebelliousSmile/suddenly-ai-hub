#!/usr/bin/env python3
"""
Session 2 - Scraping Campagnes jdRoll (Version Robuste)
Date: 2026-05-13
Objectif : Scrapper les 20 campagnes JDR de jdRoll.org
"""

import requests
import re
import os
import json
import time
import random
from datetime import datetime

# =========================================
# CONFIGURATION
# =========================================

BASE_URL = "http://www.jdroll.org"

# Headers variés
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

MIN_DELAY = 90
MAX_DELAY = 150

COOKIES_FILE = "/home/user/suddenly-ai-hub/jdroll_cookies.json"
LOGS_DIR = "/home/user/suddenly-ai-hub/logs"
DATA_DIR = "/home/user/suddenly-ai-hub/data/clean"

# =========================================
# UTILITAIRES
# =========================================

def get_delay():
    """Délai random"""
    return random.uniform(MIN_DELAY, MAX_DELAY)

def get_headers():
    """Headers variés"""
    return random.choice(HEADERS_VARIANTS)

# =========================================
# CLASSE PRINCIPALE
# =========================================

class JdRollSession2:
    """Session 2 - Scraping robuste des campagnes JDR"""
    
    def __init__(self):
        self.session = requests.Session()
        self.results = {
            "session": 2,
            "date": datetime.now().isoformat(),
            "campaigns": [],
            "metrics": {
                "total_requests": 0,
                "successful": 0,
                "failed": 0,
            },
            "errors": []
        }
        
        print("="*80)
        print("SESSION 2 - SCRAPING CAMPAGNES JDROLL (Version Robuste)")
        print("="*80)
        print(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("⏱️ Délai: 90-150 secondes entre chaque requête")
        print("📊 Volume: 20 campagnes MAX")
        print("="*80)
    
    def load_cookies(self):
        """Charger les cookies"""
        print("\n🍪 Chargement des cookies...")
        
        if not os.path.exists(COOKIES_FILE):
            print("❌ Cookies non trouvés")
            return False
        
        try:
            with open(COOKIES_FILE, "r") as f:
                cookies_dict = json.load(f)
            
            for name, value in cookies_dict.items():
                self.session.cookies.set(name, value, domain="jdroll.org")
            
            print(f"✅ {len(cookies_dict)} cookies chargés")
            return True
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def extract_campaigns_from_html(self, html):
        """Extraire la liste des campagnes du HTML"""
        print("\n📁 Extraction des campagnes...")
        
        # Regex simple et robuste
        pattern = r'href=["\'](/campagne/(\d+))["\']'
        matches = re.findall(pattern, html)
        
        campaigns = []
        for link, id_ in matches:
            # Extraire le titre (simplifié)
            title = "?"
            desc = "?"
            author = "Inconnu"
            universe = "Inconnu"
            system = "Inconnu"
            
            # Chercher autour du lien pour extraire le titre
            pos = html.find(link)
            if pos >= 0:
                # Titre: entre </a> avant le lien ou dans le contexte
                title_match = re.search(
                    rf'href=["\']\Q{link}\E[^"\']*["\']>([^<]+)</a>',
                    html,
                    re.IGNORECASE
                )
                if title_match:
                    title = title_match.group(1).strip()
                
                # Description: paragraphe suivant
                desc_match = re.search(
                    rf'href=["\']\Q{link}\E[^"\']*["\'].*?<p>(.*?)</p>',
                    html,
                    re.IGNORECASE | re.DOTALL
                )
                if desc_match:
                    desc = desc_match.group(1).strip()
                    # Nettoyer HTML
                    desc = re.sub(r'<[^>]+>', ' ', desc)
                    desc = ' '.join(desc.split())[:200]
                
                # Auteur
                author_match = re.search(
                    rf'href=["\']\Q{link}\E[^"\']*["\'].*?<i>Proposé par (.*?)</i>',
                    html,
                    re.IGNORECASE | re.DOTALL
                )
                if author_match:
                    author = author_match.group(1).strip()
                
                # Univers et système
                if 'avec le système' in desc:
                    parts = desc.split('avec le système')
                    universe = parts[0].replace('Dans l\'univers de ', '').strip()
                    system = parts[1].split('.')[0].strip() + "."
            
            campaigns.append({
                "id": id_,
                "link": link,
                "title": title,
                "universe": universe,
                "system": system,
                "author": author,
                "description": desc,
                "extracted": True
            })
        
        print(f"✅ {len(campaigns)} campagnes extraites")
        return campaigns
    
    def make_request(self, url, request_num, name=""):
        """Faire une requête sécurisée"""
        print(f"\n📋 [{request_num}/20] {name}")
        print(f"   URL: {url[:80]}")
        
        delay = get_delay()
        print(f"   Attente: {delay:.1f}s")
        
        time.sleep(delay)
        
        try:
            headers = get_headers()
            response = self.session.get(url, headers=headers, timeout=30)
            
            self.results["metrics"]["total_requests"] += 1
            
            if response.status_code == 200:
                self.results["metrics"]["successful"] += 1
                print(f"✅ {response.status_code} - {len(response.text)} caractères")
                return response
            else:
                self.results["metrics"]["failed"] += 1
                print(f"⚠️ {response.status_code}")
                return None
                
        except Exception as e:
            self.results["metrics"]["failed"] += 1
            self.results["errors"].append(f"Requête {request_num}: {str(e)}")
            print(f"❌ Erreur: {e}")
            return None
    
    def scrape_campaign(self, campaign, request_num):
        """Scrapper une campagne spécifique"""
        try:
            response = self.make_request(
                f"{BASE_URL}{campaign['link']}",
                request_num=request_num,
                name=f"Campagne #{campaign['id']}"
            )
            
            if response:
                campaign["html_size"] = len(response.text)
                campaign["scraped"] = True
                campaign["scraped_at"] = datetime.now().isoformat()
                
                # Sauvegarder HTML
                html_file = f"data/clean/campaign_{campaign['id']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                os.makedirs(os.path.dirname(html_file), exist_ok=True)
                with open(html_file, "w", encoding="utf-8") as f:
                    f.write(response.text)
                
                campaign["html_file"] = html_file
                print(f"   HTML sauvegardé: {html_file}")
                
        except Exception as e:
            self.results["errors"].append(f"Campagne {campaign['id']}: {str(e)}")
            campaign["error"] = str(e)
    
    def save_results(self):
        """Sauvegarder les résultats"""
        print("\n" + "="*80)
        print("💾 Sauvegarde des résultats...")
        
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = f"{LOGS_DIR}/session2_{timestamp}.json"
        data_file = f"{DATA_DIR}/campagnes_{timestamp}.json"
        
        # Calculer le taux de succès
        total = self.results["metrics"]["total_requests"]
        if total > 0:
            success_rate = (self.results["metrics"]["successful"] / total * 100)
            self.results["metrics"]["success_rate"] = f"{success_rate:.1f}%"
        else:
            self.results["metrics"]["success_rate"] = "N/A"
        
        # Sauvegarder
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(self.results, f, indent=2, ensure_ascii=False)
        
        with open(data_file, "w", encoding="utf-8") as f:
            json.dump(self.results["campaigns"], f, indent=2, ensure_ascii=False)
        
        print(f"✅ Log: {log_file}")
        print(f"✅ Données: {data_file}")
        
        # Résumé
        print("\n" + "="*80)
        print("📊 SYNTHÈSE SESSION 2")
        print("="*80)
        print(f"Requêtes: {self.results['metrics']['total_requests']}/20")
        print(f"Succès: {self.results['metrics']['successful']} ({self.results['metrics'].get('success_rate', 'N/A')})")
        print(f"Échecs: {self.results['metrics']['failed']}")
        print(f"Campagnes scrapées: {len(self.results['campaigns'])}")
        
        if self.results["errors"]:
            print(f"\n⚠️ Erreurs: {len(self.results['errors'])}")
            for err in self.results["errors"][:3]:
                print(f"   • {err}")
        
        return log_file, data_file
    
    def run(self):
        """Exécuter la session"""
        print("\n🚀 LANCEMENT SESSION 2")
        
        # 1. Charger cookies
        if not self.load_cookies():
            print("❌ Impossible de continuer")
            return
        
        # 2. Scraper la page d'accueil
        print("\n" + "="*80)
        print("📋 EXTRACTION PAGE D'ACCUEIL")
        print("="*80)
        
        response = self.make_request(
            f"{BASE_URL}/",
            request_num=1,
            name="Page d'accueil"
        )
        
        if not response or response.status_code != 200:
            print("❌ Échec de la page d'accueil")
            return
        
        # Extraire les campagnes
        self.results["campaigns"] = self.extract_campaigns_from_html(response.text)
        
        if not self.results["campaigns"]:
            print("❌ Aucune campagne trouvée")
            return
        
        print(f"\n✅ {len(self.results['campaigns'])} campagnes prêtes à scraper")
        for camp in self.results["campaigns"][:5]:
            print(f"   • f={camp['id']}: {camp['title'][:50]}")
        
        # 3. Scraper chaque campagne
        print("\n" + "="*80)
        print("📋 SCRAPPING DES CAMPAGNES")
        print("="*80)
        
        for i, campaign in enumerate(self.results["campaigns"], 2):
            self.scrape_campaign(campaign, request_num=i)
        
        # 4. Sauvegarder
        log_file, data_file = self.save_results()
        
        print("\n" + "="*80)
        print("✅ SESSION 2 TERMINÉE")
        print("="*80)

# =========================================
# POINT D'ENTRÉE
# =========================================

if __name__ == "__main__":
    session = JdRollSession2()
    session.run()
