#!/usr/bin/env python3
"""
Scraping de La Cour d'Obéron avec authentification
==================================================
Ce script se connecte à La Cour d'Obéron et scrap les archives publiques.
"""

import requests
import json
import time
import re
from datetime import datetime
from typing import List, Dict
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv('/home/user/suddenly-muses/.env')

class LaCourOberonScraper:
    """Scraper pour La Cour d'Obéron avec authentification"""
    
    def __init__(self, base_url: str = "https://couroberon.com/Salons"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Headers réalistes
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
        }
        
        self.session.headers.update(self.headers)
        
    def login(self, username: str, password: str) -> bool:
        """
        Se connecter au forum
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            
        Returns:
            True si succès, False sinon
        """
        print("🔐 Tentative de connexion...")
        
        # Page de login
        login_url = f"{self.base_url}/ucp.php?mode=login"
        
        try:
            # Récupérer le formulaire de login
            response = self.session.get(login_url, timeout=10)
            
            if response.status_code != 200:
                print(f"❌ Erreur accès login: {response.status_code}")
                return False
            
            # Extraire les tokens CSRF
            csrf_token = re.search(r'name="csrf_token" value="([^"]+)"', response.text)
            redirect = re.search(r'name="redirect" value="([^"]+)"', response.text)
            
            if csrf_token:
                csrf_token = csrf_token.group(1)
            else:
                print("⚠️ Token CSRF non trouvé")
                csrf_token = ""
            
            if redirect:
                redirect = redirect.group(1)
            else:
                redirect = ""
            
            # Données du formulaire
            login_data = {
                "username": username,
                "password": password,
                "autologin": "off",
                "redirect": redirect or "",
                "submit": "Se connecter",
            }
            
            if csrf_token:
                login_data["csrf_token"] = csrf_token
            
            # Soumettre le login
            login_response = self.session.post(
                login_url,
                data=login_data,
                timeout=10
            )
            
            # Vérifier la connexion
            if "logout" in login_response.text or "/ucp.php?mode=logout" in login_response.text:
                print("✅ Connexion réussie!")
                
                # Sauvegarder les cookies
                self.save_cookies()
                return True
            else:
                print("❌ Échec de la connexion")
                print(f"Status: {login_response.status_code}")
                return False
                
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            return False
    
    def save_cookies(self):
        """Sauvegarder les cookies dans .env"""
        import os
        
        cookies_dict = {}
        for cookie in self.session.cookies:
            cookies_dict[cookie.name] = cookie.value
        
        # Créer le fichier de cookies
        cookies_file = "/home/user/suddenly-muses/cookies.json"
        with open(cookies_file, "w") as f:
            json.dump(cookies_dict, f)
        
        print(f"💾 Cookies sauvegardés: {cookies_file}")
    
    def load_cookies(self):
        """Charger les cookies depuis le fichier"""
        import os
        
        cookies_file = "/home/user/suddenly-muses/cookies.json"
        
        if os.path.exists(cookies_file):
            with open(cookies_file, "r") as f:
                cookies_dict = json.load(f)
            
            for name, value in cookies_dict.items():
                self.session.cookies.set(name, value, domain="couroberon.com")
            
            print("✅ Cookies chargés")
            return True
        else:
            print("⚠️ Aucun fichier de cookies trouvé")
            return False
    
    def get_forum_list(self) -> List[int]:
        """Récupérer la liste des forums accessibles"""
        print("\n📁 Récupération de la liste des forums...")
        
        forums = []
        
        # Essayer les forums 1-20
        for forum_id in range(1, 21):
            forum_url = f"{self.base_url}/viewforum.php?f={forum_id}"
            
            try:
                response = self.session.get(forum_url, timeout=10)
                
                if response.status_code == 200 and "viewforum" in response.url:
                    # Vérifier si le forum contient des topics
                    if "viewtopic" in response.text or "topic" in response.text.lower():
                        forums.append(forum_id)
                        print(f"   ✅ Forum {forum_id} accessible")
                elif response.status_code == 503:
                    print(f"   ⚠️ Forum {forum_id}: 503 (Cloudflare?)")
                # else:
                #     print(f"   ❌ Forum {forum_id}: {response.status_code}")
                    
            except Exception as e:
                print(f"   ❌ Forum {forum_id}: {e}")
            
            time.sleep(3)  # Respect du délai
        
        return forums
    
    def get_topics_from_forum(self, forum_id: int) -> List[Dict]:
        """Récupérer les topics d'un forum"""
        print(f"\n📝 Récupération des topics du forum {forum_id}...")
        
        topics = []
        forum_url = f"{self.base_url}/viewforum.php?f={forum_id}"
        
        try:
            response = self.session.get(forum_url, timeout=15)
            
            if response.status_code != 200:
                print(f"   ❌ Erreur: {response.status_code}")
                return topics
            
            html = response.text
            
            # Extraire les topics
            topic_pattern = r'<a[^>]+href=["\'](/Salons/viewtopic\.php[^"\']+)["\'][^>]*>(.*?)</a>'
            links = re.findall(topic_pattern, html, re.IGNORECASE | re.DOTALL)
            
            for url, title in links:
                # Extraire l'ID
                match = re.search(r't=(\d+)', url)
                topic_id = match.group(1) if match else None
                
                if topic_id:
                    # Nettoyer le titre
                    title_clean = re.sub(r'<[^>]+>', '', title).strip()
                    
                    topics.append({
                        "id": topic_id,
                        "title": title_clean,
                        "url": f"{self.base_url}{url}",
                        "forum_id": forum_id
                    })
            
            print(f"   ✅ {len(topics)} topics trouvés")
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
        
        time.sleep(3)
        return topics
    
    def scrape_topic(self, topic_url: str) -> Dict:
        """Scrapper un sujet individuel"""
        print(f"\n🗨️ Scraper le topic: {topic_url[:80]}...")
        
        try:
            response = self.session.get(topic_url, timeout=15)
            
            if response.status_code != 200:
                print(f"   ❌ Erreur: {response.status_code}")
                return None
            
            html = response.text
            
            # Extraire le titre
            title_match = re.search(r'<h1[^>]*>(.*?)</h1>', html, re.DOTALL)
            title = re.sub(r'<[^>]+>', '', title_match.group(1)).strip() if title_match else "N/A"
            
            # Extraire les posts
            posts = []
            
            # Les posts sont souvent dans des divs avec class "post"
            post_pattern = r'<div[^>]*class=["\']post[^"\']*["\'][^>]*>(.*?)</div>'
            post_divs = re.findall(post_pattern, html, re.DOTALL)
            
            for post_div in post_divs:
                # Extraire le nom d'auteur
                author_match = re.search(r'class=["\']post-author[^"\']*["\'][^>]*>([^<]+)</a>', post_div)
                author = author_match.group(1).strip() if author_match else "Anonyme"
                
                # Anonymiser le nom
                author_anon = f"Author_{len(posts)+1}"
                
                # Extraire le contenu du message
                content_match = re.search(r'class=["\']post-content[^"\']*["\'][^>]*>(.*?)</div>', post_div, re.DOTALL)
                content = re.sub(r'<[^>]+>', ' ', content_match.group(1)).strip() if content_match else ""
                
                # Nettoyer le contenu
                content = re.sub(r'\s+', ' ', content).strip()
                
                # Extraire la date si possible
                date_match = re.search(r'datetime=["\']([^"\']+)["\']', post_div)
                date = date_match.group(1) if date_match else datetime.now().isoformat()
                
                posts.append({
                    "author": author_anon,
                    "original_author": author,
                    "content": content,
                    "date": date
                })
            
            # Extraire l'ID du topic
            topic_id_match = re.search(r't=(\d+)', topic_url)
            topic_id = topic_id_match.group(1) if topic_id_match else None
            
            result = {
                "topic_id": topic_id,
                "title": title,
                "url": topic_url,
                "posts": posts,
                "scraped_at": datetime.now().isoformat()
            }
            
            print(f"   ✅ Topic scrapé: {len(posts)} posts")
            
            return result
            
        except Exception as e:
            print(f"   ❌ Erreur: {e}")
            return None
    
    def scrape_forum(self, forum_id: int, max_topics: int = 50) -> List[Dict]:
        """Scrapper un forum entier"""
        print(f"\n🚀 Scraping du forum {forum_id}...")
        
        # Récupérer les topics
        topics = self.get_topics_from_forum(forum_id)
        
        if not topics:
            print("⚠️ Aucun topic trouvé")
            return []
        
        # Limiter le nombre de topics
        topics = topics[:min(max_topics, len(topics))]
        
        # Scrapper chaque topic
        all_data = []
        
        for i, topic in enumerate(topics, 1):
            print(f"\n[{i}/{len(topics)}] {topic['title'][:60]}...")
            
            topic_data = self.scrape_topic(topic['url'])
            
            if topic_data:
                all_data.append(topic_data)
            
            time.sleep(3)  # Respect du délai
        
        print(f"\n✅ Forum {forum_id} terminé: {len(all_data)} topics scrapés")
        return all_data
    
    def save_to_jsonl(self, data: List[Dict], filename: str):
        """Sauvegarder les données en JSONL"""
        print(f"\n💾 Sauvegarde dans {filename}...")
        
        with open(filename, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        print(f"✅ {len(data)} items sauvegardés")
    
    def scrape_all(self, max_forums: int = 3, max_topics_per_forum: int = 50):
        """Scrapper tous les forums"""
        print("🎯 DÉMARRAGE DU SCRAPING COMPLET")
        print("="*70)
        
        # Charger les cookies ou faire le login
        if not self.load_cookies():
            print("\n⚠️ Veuillez créer un compte sur La Cour d'Obéron et vous connecter d'abord")
            print("\nInstructions:")
            print("1. Va sur: https://couroberon.com/Salons/ucp.php?mode=register")
            print("2. Crée un compte")
            print("3. Connecte-toi avec ton navigateur")
            print("4. Copie les cookies (voir README)")
            return
        
        # Récupérer la liste des forums
        forums = self.get_forum_list()
        
        if not forums:
            print("⚠️ Aucun forum accessible avec ce compte")
            return
        
        print(f"\n✅ {len(forums)} forums trouvés")
        
        # Scraper chaque forum
        all_data = []
        
        for i, forum_id in enumerate(forums[:max_forums], 1):
            print(f"\n{'='*70}")
            print(f"FORUM {i}/{min(len(forums), max_forums)}: #{forum_id}")
            print("="*70)
            
            forum_data = self.scrape_forum(forum_id, max_topics_per_forum)
            all_data.extend(forum_data)
        
        # Sauvegarder
        if all_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/couroberon_{timestamp}.jsonl"
            self.save_to_jsonl(all_data, filename)
            
            print("\n" + "="*70)
            print("✅ SCRAPING TERMINÉ")
            print("="*70)
            print(f"Total topics: {len(all_data)}")
            print(f"Total posts: {sum(len(t['posts']) for t in all_data)}")
            print(f"Fichier: {filename}")


def main():
    """Point d'entrée"""
    scraper = LaCourOberonScraper()
    
    # Options
    MAX_FORUMS = 3
    MAX_TOPICS_PER_FORUM = 50
    
    # Lancer le scraping
    scraper.scrape_all(
        max_forums=MAX_FORUMS,
        max_topics_per_forum=MAX_TOPICS_PER_FORUM
    )


if __name__ == "__main__":
    main()
