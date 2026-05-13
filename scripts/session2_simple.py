#!/usr/bin/env python3
"""
Session 2 - Version ULTRA SIMPLE
Extraction minimale, robustesse maximale
"""

import requests
import re
import os
import json
import time
import random
from datetime import datetime

BASE_URL = "http://www.jdroll.org"

# Headers variés
HEADERS = [
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36"},
    {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.4 Safari/605.1.15"},
    {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36 Edg/125.0.0.0"},
]

MIN_DELAY = 90
MAX_DELAY = 150

COOKIES_FILE = "/home/user/suddenly-ai-hub/jdroll_cookies.json"
LOGS_DIR = "/home/user/suddenly-ai-hub/logs"
DATA_DIR = "/home/user/suddenly-ai-hub/data/clean"

def random_delay():
    return random.uniform(MIN_DELAY, MAX_DELAY)

def random_headers():
    return random.choice(HEADERS)

class SimpleSession2:
    def __init__(self):
        self.session = requests.Session()
        self.campaigns = []
        self.requests_count = 0
        self.success_count = 0
        self.failed_count = 0
        
        print("="*70)
        print("SESSION 2 - VERSION ULTRA SIMPLE")
        print("="*70)
        print(f"Début: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("="*70)
    
    def load_cookies(self):
        if not os.path.exists(COOKIES_FILE):
            print("❌ Cookies non trouvés")
            return False
        
        try:
            with open(COOKIES_FILE) as f:
                cookies = json.load(f)
            for name, val in cookies.items():
                self.session.cookies.set(name, val, domain="jdroll.org")
            print(f"✅ {len(cookies)} cookies chargés")
            return True
        except Exception as e:
            print(f"❌ Erreur cookies: {e}")
            return False
    
    def extract_campaigns(self, html):
        """Extraction ultra-simple des campagnes"""
        print("\n📁 Extraction...")
        
        try:
            # Regex très simple
            pattern = r'href=["\'](/campagne/(\d+))["\']'
            matches = re.findall(pattern, html)
            
            self.campaigns = []
            for link, id_ in matches:
                self.campaigns.append({
                    "id": id_,
                    "link": link,
                    "title": f"Campagne {id_}",  # Valeur par défaut
                    "universe": "Inconnu",
                    "system": "Inconnu",
                    "author": "Inconnu",
                    "description": "",
                    "scraped": False
                })
            
            print(f"✅ {len(self.campaigns)} campagnes extraites")
            return True
        except Exception as e:
            print(f"❌ Erreur extraction: {e}")
            return False
    
    def make_request(self, url, name):
        """Requête ultra-simple"""
        delay = random_delay()
        print(f"\n📋 [{self.requests_count+1}/20] {name}")
        print(f"   Attente: {delay:.1f}s")
        
        time.sleep(delay)
        
        try:
            self.requests_count += 1
            headers = random_headers()
            resp = self.session.get(url, headers=headers, timeout=30)
            
            if resp.status_code == 200:
                self.success_count += 1
                print(f"✅ {resp.status_code} - {len(resp.text)} octets")
                return resp
            else:
                self.failed_count += 1
                print(f"⚠️ {resp.status_code}")
                return None
        except Exception as e:
            self.failed_count += 1
            print(f"❌ {e}")
            return None
    
    def scrape_campaign(self, camp):
        """Scraping d'une campagne"""
        try:
            resp = self.make_request(
                f"{BASE_URL}{camp['link']}",
                f"Campagne #{camp['id']}"
            )
            
            if resp:
                camp["html_size"] = len(resp.text)
                camp["scraped"] = True
                
                # Sauvegarder HTML
                os.makedirs(DATA_DIR, exist_ok=True)
                html_file = f"data/clean/camp_{camp['id']}_{datetime.now().strftime('%Y%m%d')}.html"
                with open(html_file, 'w', encoding='utf-8') as f:
                    f.write(resp.text)
                camp["html_file"] = html_file
                
                # EXTRACTION SIMPLE
                html = resp.text
                
                # Titre
                try:
                    title_match = re.search(r'<h4[^>]*>([^<]+)</h4>', html)
                    if title_match:
                        camp["title"] = title_match.group(1).strip()
                except:
                    pass
                
                # Description
                try:
                    desc_match = re.search(r'<p[^>]*>([^<]+)</p>', html)
                    if desc_match:
                        desc = desc_match.group(1).strip()
                        desc = re.sub(r'<[^>]+>', ' ', desc)
                        camp["description"] = desc[:200]
                except:
                    pass
                
                # Univers et système
                try:
                    if 'avec le système' in camp["description"]:
                        parts = camp["description"].split('avec le système')
                        camp["universe"] = parts[0].replace('Dans l\'univers de ', '').strip()
                        camp["system"] = parts[1].split('.')[0].strip() + "."
                except:
                    pass
                
        except Exception as e:
            camp["error"] = str(e)
    
    def save_results(self):
        """Sauvegarder"""
        print("\n" + "="*70)
        print("💾 Sauvegarde...")
        
        os.makedirs(LOGS_DIR, exist_ok=True)
        os.makedirs(DATA_DIR, exist_ok=True)
        
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Log
        log_data = {
            "session": 2,
            "date": datetime.now().isoformat(),
            "metrics": {
                "total": self.requests_count,
                "success": self.success_count,
                "failed": self.failed_count,
                "rate": f"{(self.success_count/max(self.requests_count,1)*100):.1f}%"
            },
            "campaigns_scraped": len([c for c in self.campaigns if c.get("scraped")])
        }
        
        log_file = f"{LOGS_DIR}/session2_{ts}.log"
        with open(log_file, 'w', encoding='utf-8') as f:
            f.write(json.dumps(log_data, indent=2, ensure_ascii=False))
        
        # Données
        data_file = f"{DATA_DIR}/campagnes_{ts}.json"
        with open(data_file, 'w', encoding='utf-8') as f:
            json.dump(self.campaigns, f, indent=2, ensure_ascii=False)
        
        print(f"✅ Log: {log_file}")
        print(f"✅ Données: {data_file}")
        
        # Résumé
        print("\n" + "="*70)
        print("📊 SYNTHÈSE")
        print("="*70)
        print(f"Requêtes: {self.requests_count}/20")
        print(f"Succès: {self.success_count}")
        print(f"Échecs: {self.failed_count}")
        print(f"Campagnes scrapées: {len([c for c in self.campaigns if c.get('scraped')])}")
        print("="*70)
        
        return log_file, data_file
    
    def run(self):
        # 1. Cookies
        if not self.load_cookies():
            return
        
        # 2. Page d'accueil
        resp = self.make_request(f"{BASE_URL}/", "Page d'accueil")
        if not resp:
            print("❌ Page d'accueil échouée")
            return
        
        # 3. Extraire campagnes
        if not self.extract_campaigns(resp.text):
            return
        
        # 4. Scraper chaque campagne
        for camp in self.campaigns:
            self.scrape_campaign(camp)
        
        # 5. Sauvegarder
        self.save_results()

if __name__ == "__main__":
    SimpleSession2().run()
