#!/usr/bin/env python3
"""
Scraping jdRoll - Extrême prudence
==================================
Délais longs, volume minimal
"""

import requests
import time
import re
import os
import json
from datetime import datetime

print("🐌 Démarrage...")
print("⏱️ Délai: 60 secondes")
print("="*70)

BASE_URL = "http://www.jdroll.org"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
    "Accept-Language": "fr-FR,fr;q=0.9",
}

# Charger cookies
cookies_file = "/home/user/suddenly-ai-hub/jdroll_cookies.json"
session = requests.Session()

if os.path.exists(cookies_file):
    with open(cookies_file, "r") as f:
        cookies_dict = json.load(f)
    for name, value in cookies_dict.items():
        session.cookies.set(name, value, domain="jdroll.org")
    print("✅ Cookies chargés")
else:
    print("❌ Pas de cookies")
    exit()

time.sleep(60)  # 1 minute avant première requête

print("\n📋 1/1 - Page d'accueil...")
resp = session.get(BASE_URL, headers=HEADERS, timeout=15)
print(f"Status: {resp.status_code}")

if resp.status_code == 200:
    print("✅ Page accessible")
    time.sleep(60)
    
    # Extraire forums
    forums = re.findall(r'href=["\'](/viewforum\.php[^"\']*)["\']', resp.text)
    print(f"Forums trouvés: {len(forums)}")
    
    if forums:
        print("\n📁 Forums:")
        for i, link in enumerate(forums[:1], 1):  # 1 forum seulement
            match = re.search(r'f=(\d+)', link)
            forum_id = match.group(1) if match else "?"
            print(f"   {i}. f={forum_id}")
            
            # Test forum
            time.sleep(60)
            forum_resp = session.get(f"{BASE_URL}{link}", headers=HEADERS, timeout=15)
            print(f"   Status: {forum_resp.status_code}")
            
            if forum_resp.status_code == 200:
                # Extraire 1 topic max
                topics = re.findall(r'viewtopic\.php[^"\']*', forum_resp.text)
                print(f"   Topics: {len(topics)}")
                
                if topics:
                    time.sleep(60)
                    topic_url = f"{BASE_URL}/viewtopic.php{topics[0]}"
                    topic_resp = session.get(topic_url, headers=HEADERS, timeout=15)
                    print(f"   Topic status: {topic_resp.status_code}")
                    
                    if topic_resp.status_code == 200:
                        posts = re.findall(r'post', topic_resp.text, re.IGNORECASE)
                        print(f"   Posts détectés: {len(posts)}")
                        
                        # Sauvegarder
                        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"data/jdroll_test_{timestamp}.json"
                        
                        data = {
                            "scraped_at": datetime.now().isoformat(),
                            "forum_id": forum_id,
                            "topic_id": "test",
                            "posts_found": len(posts)
                        }
                        
                        with open(filename, "w") as f:
                            json.dump(data, f, indent=2)
                        
                        print(f"\n✅ Test terminé - données sauvegardées dans {filename}")

print("\n" + "="*70)
print("✅ SCRAPING TERMINÉ - EN ATTENTE NOUVELLE INSTRUCTION")
print("="*70)
