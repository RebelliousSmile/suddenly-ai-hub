#!/usr/bin/env python3
"""
Scraping jdRoll avec authentification
======================================
Ce script se connecte à jdRoll.org et scrap les discussions RP.

⚠️ SÉCURITÉ: Ne jamais committer les identifiants ou cookies !
"""

import requests
import json
import time
import re
import os
from datetime import datetime
from typing import List, Dict
from getpass import getpass
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv('/home/user/suddenly-muses/.env')

class JdRollScraper:
    """Scraper pour jdRoll avec authentification"""
    
    def __init__(self, base_url: str = "http://www.jdroll.org"):
        self.base_url = base_url
        self.session = requests.Session()
        
        # Headers réalistes
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "fr-FR,fr;q=0.9,en;q=0.8",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
            "Upgrade-Insecure-Requests": "1",
            "Origin": base_url,
        }
        
        self.session.headers.update(self.headers)
        
    def login_with_credentials(self, username: str, password: str) -> bool:
        """
        Se connecter avec login/mot de passe
        
        Args:
            username: Nom d'utilisateur
            password: Mot de passe
            
        Returns:
            True si succès, False sinon
        """
        print("🔐 Tentative de connexion...")
        
        # jdRoll utilise /login au lieu de /ucp.php?mode=login
        login_url = f"{self.base_url}/login"
        
        try:
            # Récupérer le formulaire de login
            response = self.session.get(login_url, timeout=10)
            
            print(f"   URL finale: {response.url}")
            print(f"   Status: {response.status_code}")
            
            if response.status_code != 200:
                print(f"❌ Erreur accès login: {response.status_code}")
                return False
            
            # jdRoll n'utilise pas de CSRF token, on va directement poster
            # Données du formulaire
            login_data = {
                "auth_key": "",  # jdRoll n'en utilise pas
                "username": username,
                "password": password,
            }
            
            # Soumettre le login
            login_response = self.session.post(
                login_url,
                data=login_data,
                timeout=10
            )
            
            print(f"   Response URL: {login_response.url}")
            
            # Vérifier la connexion - jdRoll redirige vers / après login
            # ou affiche un lien de déconnexion
            if "logout" in login_response.text.lower() or "/logout" in login_response.url:
                print("✅ Connexion réussie!")
                
                # Sauvegarder les cookies
                self.save_cookies()
                return True
            elif response.status_code == 302 or response.url.startswith(self.base_url):
                # Vérifier si on a accès aux pages membres
                test_response = self.session.get(f"{self.base_url}/", timeout=10)
                if "logout" in test_response.text.lower():
                    print("✅ Connexion réussie (vérification)!")
                    self.save_cookies()
                    return True
            
            print("❌ Échec de la connexion")
            print(f"Status: {login_response.status_code}")
            print(f"URL finale: {login_response.url}")
            return False
                
        except Exception as e:
            print(f"❌ Erreur de connexion: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def login_with_cookies(self, cookies_file: str = None) -> bool:
        """
        Se connecter avec des cookies sauvegardés
        
        Args:
            cookies_file: Chemin vers le fichier de cookies
            
        Returns:
            True si succès, False sinon
        """
        print("🔐 Chargement des cookies...")
        
        if cookies_file is None:
            cookies_file = "/home/user/suddenly-muses/jdroll_cookies.json"
        
        if not os.path.exists(cookies_file):
            print(f"⚠️ Fichier de cookies non trouvé: {cookies_file}")
            print("   Lancez le script avec --login pour authentifier")
            return False
        
        try:
            with open(cookies_file, "r") as f:
                cookies_dict = json.load(f)
            
            for name, value in cookies_dict.items():
                self.session.cookies.set(name, value, domain="jdroll.org")
            
            # Tester la connexion
            test_response = self.session.get(f"{self.base_url}/", timeout=10)
            
            if "logout" in test_response.text:
                print("✅ Cookies valides!")
                return True
            else:
                print("⚠️ Cookies expirés ou invalides")
                return False
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
            return False
    
    def save_cookies(self, cookies_file: str = None):
        """Sauvegarder les cookies"""
        if cookies_file is None:
            cookies_file = "/home/user/suddenly-muses/jdroll_cookies.json"
        
        cookies_dict = {}
        for cookie in self.session.cookies:
            cookies_dict[cookie.name] = cookie.value
        
        with open(cookies_file, "w") as f:
            json.dump(cookies_dict, f)
        
        print(f"💾 Cookies sauvegardés: {cookies_file}")
    
    def get_forums(self) -> List[Dict]:
        """Récupérer la liste des forums"""
        print("\n📁 Récupération de la liste des forums...")
        
        forums = []
        forum_url = f"{self.base_url}/index.php"
        
        try:
            response = self.session.get(forum_url, timeout=15)
            
            if response.status_code == 200:
                html = response.text
                
                # Extraire les forums
                forum_pattern = r'<a[^>]+href=["\'](/viewforum\.php[^"\']+)["\'][^>]*>(.*?)</a>'
                forum_links = re.findall(forum_pattern, html, re.IGNORECASE | re.DOTALL)
                
                for url, title in forum_links:
                    # Nettoyer le titre
                    title_clean = re.sub(r'<[^>]+>', '', title).strip()
                    
                    # Extraire l'ID
                    match = re.search(r'f=(\d+)', url)
                    forum_id = match.group(1) if match else None
                    
                    if forum_id:
                        forums.append({
                            "id": forum_id,
                            "title": title_clean,
                            "url": f"{self.base_url}{url}"
                        })
                
                print(f"✅ {len(forums)} forums trouvés")
                
        except Exception as e:
            print(f"❌ Erreur: {e}")
        
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
            topic_pattern = r'<a[^>]+href=["\'](/viewtopic\.php[^"\']+)["\'][^>]*>(.*?)</a>'
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
        
        time.sleep(2)
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
            
            # Pattern pour les posts phpBB
            post_pattern = r'<div[^>]*id=["\']post\d+["\'][^>]*>(.*?)</div>'
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
    
    def scrape_forum(self, forum_id: int, max_topics: int = 20) -> List[Dict]:
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
    
    def scrape_all(self, max_forums: int = 2, max_topics_per_forum: int = 20):
        """Scrapper tous les forums"""
        print("🎯 DÉMARRAGE DU SCRAPING JDROLL")
        print("="*70)
        
        # Charger les cookies ou faire le login
        if not self.login_with_cookies():
            print("\n⚠️ Veuillez vous authentifier d'abord")
            print("\nOptions:")
            print("  --login: Utiliser login/mot de passe")
            print("  --cookies: Fichier de cookies personnalisé")
            return
        
        # Récupérer la liste des forums
        forums = self.get_forums()
        
        if not forums:
            print("⚠️ Aucun forum accessible avec ce compte")
            return
        
        print(f"\n✅ {len(forums)} forums trouvés")
        
        # Scraper chaque forum
        all_data = []
        
        for i, forum in enumerate(forums[:max_forums], 1):
            print(f"\n{'='*70}")
            print(f"FORUM {i}/{min(len(forums), max_forums)}: #{forum['id']} - {forum['title'][:50]}")
            print("="*70)
            
            forum_data = self.scrape_forum(forum['id'], max_topics_per_forum)
            all_data.extend(forum_data)
        
        # Sauvegarder
        if all_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"data/jdroll_{timestamp}.jsonl"
            self.save_to_jsonl(all_data, filename)
            
            print("\n" + "="*70)
            print("✅ SCRAPING TERMINÉ")
            print("="*70)
            print(f"Total topics: {len(all_data)}")
            print(f"Total posts: {sum(len(t['posts']) for t in all_data)}")
            print(f"Fichier: {filename}")


def main():
    """Point d'entrée"""
    import sys
    import argparse
    
    parser = argparse.ArgumentParser(description='Scraper jdRoll.org')
    parser.add_argument('--login', action='store_true', help='Se connecter avec login/mdp')
    parser.add_argument('--username', type=str, help='Nom d\'utilisateur')
    parser.add_argument('--password', type=str, help='Mot de passe')
    parser.add_argument('--cookies', type=str, help='Fichier de cookies')
    parser.add_argument('--max-forums', type=int, default=2, help='Nombre max de forums')
    parser.add_argument('--max-topics', type=int, default=20, help='Topics max par forum')
    
    args = parser.parse_args()
    
    scraper = JdRollScraper()
    
    # Authentification
    if args.login or not scraper.login_with_cookies(args.cookies):
        if args.username and args.password:
            scraper.login_with_credentials(args.username, args.password)
        else:
            print("\n🔐 Authentification nécessaire")
            username = input("Nom d'utilisateur: ")
            password = getpass("Mot de passe: ")
            scraper.login_with_credentials(username, password)
    elif args.cookies:
        scraper.login_with_cookies(args.cookies)
    
    # Lancer le scraping
    scraper.scrape_all(
        max_forums=args.max_forums,
        max_topics_per_forum=args.max_topics
    )


if __name__ == "__main__":
    main()
